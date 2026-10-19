"""Deterministic tests: no models, OS changes, GPUs or real databases."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rad_resources import GIB,MIB,ResourceGuard,ResourcePressureError,ResourceSnapshot,add_resource_arguments,guard_from_args,_gpu_probe,_linux_cgroups

class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def sleep(self,seconds):self.now+=seconds

def snap(t,**kw):
    d=dict(timestamp=t,ram_total_bytes=64*GIB,ram_available_bytes=48*GIB,process_rss_bytes=2*GIB,swap_in_bytes=0,swap_out_bytes=0)
    d.update(kw);return ResourceSnapshot(**d)

def host(t,**kw):
    d=dict(wsl=True,host_total_bytes=80*GIB,host_available_bytes=60*GIB,host_pagefile_in_bytes=100,host_pagefile_out_bytes=100,host_counter_time_seconds=t,host_pagefile_verified=True,host_pagefile_source='verified_test_counter',host_measurement_kind='upper_bound')
    d.update(kw);return snap(t,**d)

class ResourceTests(unittest.TestCase):
    def guard(self,samples,**kw):
        c=Clock();seq=iter(samples);d=dict(ram_budget_bytes=8*GIB,probe=lambda:next(seq),clock=c,sleeper=c.sleep);d.update(kw)
        return ResourceGuard(**d),c
    def failure(self,samples,code,**kw):
        g,_=self.guard(samples,**kw)
        with self.assertRaises(ResourcePressureError) as e:g.preflight()
        self.assertEqual(e.exception.evidence['code'],code);self.assertEqual(e.exception.reason,'resource_pressure');return e.exception
    def test_current_rss_credited_once(self):
        g,_=self.guard([snap(i,ram_total_bytes=32*GIB,ram_available_bytes=12*GIB,process_rss_bytes=6*GIB) for i in range(3)],ram_budget_bytes=10*GIB)
        r=g.preflight();self.assertEqual(r['remaining_ram_budget_bytes'],4*GIB);self.assertEqual(r['reserve_bytes'],8*GIB)
    def test_fractional_reserve(self):
        g,_=self.guard([snap(i) for i in range(3)]);self.assertEqual(g.preflight()['reserve_bytes'],.2*64*GIB)
    def test_small_environment_requires_explicit_reserve(self):
        v=[snap(i,ram_total_bytes=8*GIB,ram_available_bytes=6*GIB,process_rss_bytes=GIB) for i in range(3)]
        self.failure(v,'insufficient_ram',ram_budget_bytes=2*GIB)
        g,_=self.guard(v,ram_budget_bytes=2*GIB,reserve_gib=1,reserve_fraction=.1);self.assertEqual(g.preflight()['status'],'admitted')
    def test_effective_cgroup(self):self.failure([snap(0,cgroup_limit_bytes=10*GIB,cgroup_available_bytes=6*GIB)],'insufficient_ram')
    def test_missing_cgroup_data(self):self.failure([snap(0,cgroup_limit_bytes=10*GIB)],'cgroup_unverified')
    def test_swap_occupancy_not_paging(self):
        g,_=self.guard([snap(i,swap_in_bytes=1000*MIB,swap_out_bytes=2000*MIB,diagnostics={'swap_used_bytes':30*GIB}) for i in range(3)])
        self.assertEqual(g.preflight()['paging']['linux_or_local']['in_bytes_per_second'],0)
    def test_one_burst_is_not_sustained(self):
        g,_=self.guard([snap(0),snap(1,swap_in_bytes=2*MIB),snap(2,swap_in_bytes=2*MIB)]);self.assertEqual(g.preflight()['status'],'admitted')
    def test_two_windows_stop(self):
        e=self.failure([snap(0),snap(1,swap_in_bytes=2*MIB),snap(2,swap_in_bytes=4*MIB)],'sustained_paging');self.assertIn('does not establish',e.evidence['attribution'])
    def test_reset_is_not_zero(self):self.failure([snap(0,swap_in_bytes=10),snap(1,swap_in_bytes=2)],'invalid_paging_window')
    def test_time_validity(self):
        for t in (0,-1,301):
            with self.subTest(t=t):self.failure([snap(0),snap(t)],'invalid_paging_window')
    def test_probe_error(self):
        def broken():raise FileNotFoundError('counters missing')
        g=ResourceGuard(GIB,probe=broken)
        with self.assertRaises(ResourcePressureError) as e:g.preflight()
        self.assertEqual(e.exception.evidence['code'],'probe_failed')
    def test_unverified_host_is_unsafe(self):self.failure([host(0,host_pagefile_verified=False)],'pagefile_actual_io_unverified')
    def test_host_ram_checked(self):self.failure([host(0,host_available_bytes=8*GIB)],'insufficient_host_ram')
    def test_host_zero_upper_bound(self):
        g,_=self.guard([host(i) for i in range(3)]);r=g.preflight()['paging']['windows_host'];self.assertEqual(r['measurement_kind'],'upper_bound');self.assertEqual(r['in_bytes_per_second'],0)
    def test_host_low_upper_bound(self):
        g,_=self.guard([host(i,host_pagefile_in_bytes=i*1000) for i in range(3)]);self.assertEqual(g.preflight()['status'],'admitted')
    def test_host_high_bound_not_called_actual_swap(self):
        e=self.failure([host(i,host_pagefile_in_bytes=i*2*MIB) for i in range(3)],'host_paging_unresolved');self.assertEqual(e.evidence['paging']['windows_host']['measurement_kind'],'upper_bound')
    def test_exact_host_paging(self):self.failure([host(i,host_pagefile_in_bytes=i*2*MIB,host_measurement_kind='exact') for i in range(3)],'sustained_paging')
    def test_host_reset_and_time_reset(self):
        self.failure([host(0),host(1,host_pagefile_in_bytes=0)],'invalid_paging_window');self.failure([host(0),host(1,host_counter_time_seconds=0)],'invalid_paging_window')
    def test_gpu_current_residency_credited(self):
        g,_=self.guard([snap(i,gpu_total_bytes=16*GIB,gpu_free_bytes=2*GIB,gpu_process_bytes=6*GIB) for i in range(3)],device='cuda:0',gpu_budget_bytes=8*GIB);self.assertEqual(g.preflight()['status'],'admitted')
    def test_gpu_cannot_borrow_shared_memory(self):self.failure([snap(0,gpu_total_bytes=16*GIB,gpu_free_bytes=GIB,gpu_process_bytes=0)],'insufficient_gpu',device='cuda:0',gpu_budget_bytes=2*GIB)
    def test_wsl_gpu_shared_must_be_verified(self):self.failure([host(0,gpu_total_bytes=16*GIB,gpu_free_bytes=8*GIB,gpu_process_bytes=0)],'gpu_shared_unverified',device='cuda:0',gpu_budget_bytes=2*GIB)
    def test_existing_shared_ok_growth_stops(self):
        def v(t,shared):return host(t,gpu_total_bytes=16*GIB,gpu_free_bytes=8*GIB,gpu_process_bytes=0,gpu_shared_verified=True,gpu_shared_used_bytes=shared)
        g,c=self.guard([v(i,20*MIB) for i in range(3)]+[v(8,100*MIB)],device='cuda:0',gpu_budget_bytes=2*GIB);g.preflight();c.now=8
        with self.assertRaises(ResourcePressureError) as e:g.check()
        self.assertEqual(e.exception.evidence['code'],'gpu_shared_growth')
    def test_throttle(self):
        g,c=self.guard([snap(0),snap(1),snap(2),snap(8)]);g.preflight()
        for _ in range(100):self.assertIsNone(g.check())
        c.now=8;self.assertEqual(g.check()['status'],'checked')
    def test_short_forced_window_keeps_baseline_not_zero_rate(self):
        samples = [snap(0),snap(1),snap(2),snap(2.1,swap_in_bytes=MIB),
                   snap(3,swap_in_bytes=2*MIB)]
        g,c=self.guard(samples);g.preflight();c.now=2.1
        receipt=g.check(force=True)
        self.assertEqual(receipt['paging']['measurement_status'],'deferred')
        self.assertNotIn('linux_or_local',receipt['paging'])
        self.assertEqual(g._previous.timestamp,2)
        c.now=3;receipt=g.check(force=True)
        self.assertEqual(receipt['paging']['linux_or_local']['elapsed_seconds'],1)
        self.assertEqual(receipt['paging']['linux_or_local']['in_bytes_per_second'],2*MIB)
    def test_deferred_check_still_enforces_capacity(self):
        g,c=self.guard([snap(0),snap(1),snap(2),snap(2.1,ram_available_bytes=GIB)])
        g.preflight();c.now=2.1
        with self.assertRaises(ResourcePressureError) as error:g.check(force=True)
        self.assertEqual(error.exception.evidence['code'],'insufficient_ram')
    def test_deferred_window_preserves_active_streak(self):
        samples=[snap(0),snap(1),snap(2),snap(3,swap_out_bytes=2*MIB),
                 snap(3.1,swap_out_bytes=2*MIB),snap(4,swap_out_bytes=4*MIB)]
        g,c=self.guard(samples);g.preflight();c.now=3;g.check(force=True)
        c.now=3.1;receipt=g.check(force=True)
        self.assertEqual(receipt['paging']['retained_consecutive_active_windows']['linux_or_local'],1)
        c.now=4
        with self.assertRaises(ResourcePressureError) as error:g.check(force=True)
        self.assertEqual(error.exception.evidence['code'],'sustained_paging')
    def test_unchanged_host_sample_deferred_then_validated(self):
        samples=[host(0),host(1),host(2),host(2.1,host_counter_time_seconds=2),host(3)]
        g,c=self.guard(samples);g.preflight();c.now=2.1
        receipt=g.check(force=True)
        self.assertEqual(receipt['paging']['measurement_status'],'deferred')
        self.assertEqual(receipt['paging']['elapsed_seconds_by_domain']['windows_host'],0)
        self.assertEqual(g._previous.timestamp,2)
        c.now=3;receipt=g.check(force=True)
        self.assertEqual(receipt['paging']['windows_host']['elapsed_seconds'],1)
        self.assertEqual(receipt['paging']['windows_host']['measurement_kind'],'upper_bound')
    def test_short_host_window_keeps_both_domain_baselines(self):
        samples=[host(0),host(1),host(2),host(3,host_counter_time_seconds=2.5),host(4)]
        g,c=self.guard(samples);g.preflight();c.now=3
        self.assertEqual(g.check(force=True)['paging']['measurement_status'],'deferred')
        self.assertEqual(g._previous.timestamp,2)
        c.now=4;receipt=g.check(force=True)
        self.assertEqual(receipt['paging']['linux_or_local']['elapsed_seconds'],2)
        self.assertEqual(receipt['paging']['windows_host']['elapsed_seconds'],2)
    def test_deferred_host_still_rejects_counter_or_time_reset(self):
        for changes in ({'host_pagefile_in_bytes':0,'host_counter_time_seconds':2},
                        {'host_pagefile_in_bytes':101,'host_counter_time_seconds':2},
                        {'host_counter_time_seconds':1.9},
                        {'host_pagefile_source':'different'}):
            with self.subTest(changes=changes):
                g,c=self.guard([host(0),host(1),host(2),host(2.1,**changes)])
                g.preflight();c.now=2.1
                with self.assertRaises(ResourcePressureError):g.check(force=True)
    def test_host_provider_cannot_remain_stale_indefinitely(self):
        samples=[host(0),host(1),host(2),host(3,host_counter_time_seconds=2),
                 host(303,host_counter_time_seconds=2)]
        g,c=self.guard(samples);g.preflight();c.now=3;g.check(force=True);c.now=303
        with self.assertRaises(ResourcePressureError) as error:g.check(force=True)
        self.assertEqual(error.exception.evidence['code'],'invalid_paging_window')
    def test_short_local_window_still_rejects_counter_reset(self):
        samples=[snap(i,swap_in_bytes=100) for i in range(3)]+[snap(2.1,swap_in_bytes=99)]
        g,c=self.guard(samples);g.preflight();c.now=2.1
        with self.assertRaises(ResourcePressureError) as error:g.check(force=True)
        self.assertEqual(error.exception.evidence['code'],'invalid_paging_window')
    def test_deferred_receipt_is_persisted_without_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resource.jsonl'
            g,c=self.guard([host(0),host(1),host(2),host(2.1,host_counter_time_seconds=2)],log_path=path)
            g.preflight();c.now=2.1;g.check(force=True)
            receipt=json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(receipt['status'],'checked')
            self.assertEqual(receipt['paging']['measurement_status'],'deferred')
            self.assertNotIn('in_bytes_per_second',json.dumps(receipt['paging']))
    def test_runtime_stop_and_no_automatic_resume(self):
        g,c=self.guard([snap(0),snap(1),snap(2),snap(8,swap_out_bytes=12*MIB),snap(14,swap_out_bytes=24*MIB)]);g.preflight();c.now=8;g.check();c.now=14
        with self.assertRaises(ResourcePressureError) as e:g.check()
        self.assertEqual(e.exception.evidence['code'],'sustained_paging')
        with self.assertRaises(ResourcePressureError) as e:g.check()
        self.assertEqual(e.exception.evidence['code'],'guard_already_stopped')
    def test_budget_exceeded(self):self.failure([snap(0,process_rss_bytes=9*GIB)],'ram_budget_exceeded')
    def test_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'resource.jsonl';self.failure([snap(0,ram_available_bytes=GIB)],'insufficient_ram',log_path=p);r=json.loads(p.read_text());self.assertEqual(r['status'],'stopped');self.assertEqual(r['reason'],'resource_pressure');self.assertEqual(r['resume_policy'],'explicit_resume_after_resource_revalidation');self.assertEqual(r['snapshot']['ram_available_bytes'],GIB)
    def test_cli_overrides(self):
        a=add_resource_arguments(argparse.ArgumentParser()).parse_args(['--ram-budget-gib','12','--gpu-budget-gib','3']);g=guard_from_args(a,GIB,2*GIB,device='cuda:0');self.assertEqual(g.ram_budget_bytes,12*GIB);self.assertEqual(g.gpu_budget_bytes,3*GIB)
    def test_cuda_visible_mapping(self):
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'GPU-a,GPU-b'}),patch('rad_resources._run',side_effect=['16000, 12000','']) as r:_gpu_probe('cuda:1');self.assertIn('--id=GPU-b',r.call_args_list[0].args[0])
    def test_invalid_settings(self):
        for kw in ({'reserve_fraction':1},{'paging_grace_samples':1},{'interval_seconds':0},{'device':'cuda:0','gpu_budget_bytes':0},{'reserve_gib':float('nan')}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):ResourceGuard(GIB,**kw)
    def test_nonfinite_counters(self):
        self.failure([snap(0,swap_in_bytes=float('nan'))],'probe_failed');self.failure([host(0,host_available_bytes=float('nan'))],'probe_failed')

    def test_effective_cgroup_checks_ancestors(self):
        with tempfile.TemporaryDirectory() as directory:
            mount=Path(directory)/'cg';child=mount/'parent'/'child';child.mkdir(parents=True)
            for where,limit,used in [(mount/'parent',GIB,GIB//4),(child,2*GIB,3*GIB//2)]:
                (where/'memory.max').write_text(str(limit));(where/'memory.current').write_text(str(used))
            original=Path.read_text
            def read(path,*args,**kwargs):
                if str(path)=='/proc/self/cgroup':return '0::/parent/child\n'
                if str(path)=='/proc/self/mountinfo':return f'42 1 0:1 / {mount} rw - cgroup2 cgroup rw\n'
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',read):
                limit,available,_=_linux_cgroups(64*GIB)
            self.assertEqual(limit,GIB);self.assertEqual(available,GIB//2)
    def test_invalid_gpu_counter_is_not_zero(self):
        self.failure([snap(0,gpu_free_bytes='unknown')],'probe_failed')
    def test_check_requires_admission(self):
        g=ResourceGuard(GIB,probe=lambda:snap(0))
        with self.assertRaises(ResourcePressureError) as error:g.check()
        self.assertEqual(error.exception.evidence['code'],'not_admitted')

if __name__=='__main__':unittest.main()
