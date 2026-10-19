"""Fail-closed, read-only resource admission and monitoring for RAD.

Budgets are estimated TOTAL task peaks, not future allocations. Current process
RSS/exact GPU residency is credited once when calculating remaining allocation.
Unattributed device-used GPU upper bounds never receive this credit.
Workers must be included in the peak estimate; only the main process is credited,
which is conservative. Swap occupancy is never used as evidence of paging.

The guard raises only in its caller; it never kills processes, changes swap,
clears caches, changes WSL limits, or resumes an interrupted experiment.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

GIB = 1024 ** 3
MIB = 1024 ** 2


class ResourcePressureError(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}
        self.reason = "resource_pressure"


@dataclass
class ResourceSnapshot:
    timestamp: float
    ram_total_bytes: int
    ram_available_bytes: int
    process_rss_bytes: int
    swap_in_bytes: int
    swap_out_bytes: int
    paging_source: str = "injected_actual_swap_counters"
    cgroup_limit_bytes: Optional[int] = None
    cgroup_available_bytes: Optional[int] = None
    wsl: bool = False
    host_total_bytes: Optional[int] = None
    host_available_bytes: Optional[int] = None
    host_pagefile_in_bytes: Optional[int] = None
    host_pagefile_out_bytes: Optional[int] = None
    host_counter_time_seconds: Optional[float] = None
    host_pagefile_verified: bool = False
    host_pagefile_source: Optional[str] = None
    host_measurement_kind: str = "exact"
    gpu_total_bytes: Optional[int] = None
    gpu_free_bytes: Optional[int] = None
    gpu_process_bytes: Optional[int] = None
    gpu_process_measurement_kind: str = "exact"
    gpu_process_source: Optional[str] = "injected_probe"
    gpu_shared_used_bytes: Optional[int] = None
    gpu_shared_verified: bool = False
    processes: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _run(argv, timeout=8):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=True)
    return result.stdout.strip()


def _read_numbers(path):
    return {parts[0].rstrip(":"): int(parts[1]) for line in Path(path).read_text().splitlines()
            if len(parts := line.split()) >= 2 and parts[1].isdigit()}


def _linux_cgroups(total):
    """Inspect effective memory limits at every visible cgroup ancestor."""
    memberships = []
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, path = line.split(':', 2)
        if not controllers or 'memory' in controllers.split(','):
            memberships.append((not controllers, path))
    constraints = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before, after = line.split(' - ', 1)
        fields, fs = before.split(), after.split()
        v2 = fs[0] == 'cgroup2'
        if not v2 and not (fs[0] == 'cgroup' and 'memory' in fs[2].split(',')):
            continue
        root, mount = fields[3], Path(fields[4].replace('\\040', ' '))
        for member_v2, member in memberships:
            if member_v2 != v2:
                continue
            # Cgroup namespaces commonly report '/' relative to the mount root.
            relative = member[len(root):] if root != '/' and member.startswith(root + '/') else member
            if member == root:
                relative = '/'
            current = mount / relative.lstrip('/')
            if not current.exists() and member == '/':
                current = mount
            if not current.exists():
                raise ValueError('Cannot resolve effective memory cgroup membership')
            while True:
                limit_path = current / ('memory.max' if v2 else 'memory.limit_in_bytes')
                used_path = current / ('memory.current' if v2 else 'memory.usage_in_bytes')
                if limit_path.exists():
                    limit_text = limit_path.read_text().strip()
                    if limit_text != 'max':
                        limit = int(limit_text)
                        # v1 uses enormous integers for unlimited; host RAM remains the cap.
                        if limit < total:
                            used = int(used_path.read_text().strip())
                            constraints.append((limit, max(0, limit - used), str(current)))
                if current == mount:
                    break
                if mount not in current.parents:
                    raise ValueError('Cgroup path escaped its mount')
                current = current.parent
    if not constraints:
        return None, None, []
    return min(x[0] for x in constraints), min(x[1] for x in constraints), constraints


_WINDOWS_DIAGNOSTICS = r'''
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
$o = Get-CimInstance Win32_OperatingSystem
$m = Get-CimInstance Win32_PerfRawData_PerfOS_Memory
foreach ($field in @('PagesInputPersec','PagesOutputPersec','PageReadsPersec','PageWritesPersec','Timestamp_PerfTime','Frequency_PerfTime')) {
 if ($null -eq $m.$field) { throw "Missing Windows memory performance counter: $field" }
}
if ([double]$m.Frequency_PerfTime -le 0) { throw 'Invalid Windows performance-counter frequency' }
$g = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory -ErrorAction SilentlyContinue)
$p = Get-CimInstance Win32_Process | Sort-Object WorkingSetSize -Descending | Select-Object -First 8 ProcessId,Name,WorkingSetSize
[pscustomobject]@{
 total_bytes = [long]$o.TotalVisibleMemorySize * 1024
 available_bytes = [long]$o.FreePhysicalMemory * 1024
 hard_pages_input = [long]$m.PagesInputPersec
 hard_pages_output = [long]$m.PagesOutputPersec
 page_read_operations = [long]$m.PageReadsPersec
 page_write_operations = [long]$m.PageWritesPersec
 raw_time = [double]$m.Timestamp_PerfTime
 raw_frequency = [double]$m.Frequency_PerfTime
 page_size = [Environment]::SystemPageSize
 gpu_adapters = @($g | Select-Object Name,SharedUsage,DedicatedUsage)
 processes = @($p)
} | ConvertTo-Json -Depth 4 -Compress
'''


def _windows_host():
    """Read host RAM/hard-fault diagnostics. These are NOT pagefile-only I/O.

    Windows' standard Memory/Page Reads counters include mapped-file hard faults.
    Their cumulative disk-page transfers are an upper bound on pagefile I/O.
    Zero/low deltas can certify zero/low actual pagefile I/O, without claiming
    exact attribution. High sustained deltas remain unresolved and stop work.
    An authoritative collector may instead supply exact counters via probe.
    """
    text = _run(['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive',
                 '-Command', _WINDOWS_DIAGNOSTICS])
    return json.loads(text.lstrip('\ufeff'))


def _gpu_probe(device):
    """Return dedicated capacity and exact own usage or a whole-device upper bound.

    WSL/WDDM may omit a process or report its memory as N/A. Device total/free
    remain independently checked; their difference bounds own residency but
    also includes contexts and other processes. Missing process data is not zero.
    """
    match = re.fullmatch(r'cuda(?::(\d+))?', str(device))
    if not match:
        raise ValueError('Only CPU and NVIDIA CUDA devices have a verified default resource probe')
    ordinal = int(match.group(1) or 0)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    physical = str(ordinal)
    if visible is not None:
        tokens = [x.strip() for x in visible.split(',') if x.strip()]
        if ordinal >= len(tokens) or tokens[ordinal] == '-1':
            raise ValueError('CUDA device is not present in CUDA_VISIBLE_DEVICES')
        physical = tokens[ordinal]
    row = _run(['nvidia-smi', '--id=' + physical, '--query-gpu=memory.total,memory.free',
                '--format=csv,noheader,nounits']).splitlines()
    if len(row) != 1:
        raise ValueError('GPU query did not identify exactly one selected device')
    total, free = [int(v.strip()) * MIB for v in row[0].split(',')]
    if total <= 0 or not 0 <= free <= total:
        raise ValueError('GPU dedicated total/free counters are inconsistent')
    process_rows = _run(['nvidia-smi', '--id=' + physical,
                         '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'])
    own = 0
    own_rows = 0
    all_own_numeric = True
    for line in process_rows.splitlines():
        pid, amount = [v.strip() for v in line.split(',')]
        if int(pid) == os.getpid():
            own_rows += 1
            if not amount.isdigit():
                all_own_numeric = False
            else:
                own += int(amount) * MIB
    if own_rows and all_own_numeric:
        if own > total:
            raise ValueError('Current-process GPU residency exceeds device capacity')
        return total, free, own, 'exact'
    return total, free, total - free, 'device_used_upper_bound'


class SystemProbe:
    """Bounded system reads; invoked at most once per configured check interval."""
    def __init__(self, device='cpu', clock=time.monotonic):
        self.device, self.clock = str(device), clock

    def __call__(self):
        system = platform.system()
        diagnostics = {}
        if system == 'Linux':
            mem = _read_numbers('/proc/meminfo')
            vm = _read_numbers('/proc/vmstat')
            status = _read_numbers('/proc/self/status')
            total, available, rss = mem['MemTotal'] * 1024, mem['MemAvailable'] * 1024, status['VmRSS'] * 1024
            page = os.sysconf('SC_PAGE_SIZE')
            sin, sout = vm['pswpin'] * page, vm['pswpout'] * page
            limit, cgavailable, constraints = _linux_cgroups(total)
            diagnostics['cgroup_constraints'] = constraints
            pressure = Path('/proc/pressure/memory')
            if pressure.exists():
                diagnostics['memory_pressure_psi'] = pressure.read_text().strip()
            diagnostics['major_page_faults_diagnostic_only'] = vm.get('pgmajfault')
            wsl = 'microsoft' in Path('/proc/sys/kernel/osrelease').read_text().lower()
            processes = []
            for count, entry in enumerate(Path('/proc').iterdir()):
                if count > 16384:
                    raise ValueError('Process inventory exceeds bounded probe capacity')
                if entry.name.isdigit():
                    try:
                        data = _read_numbers(entry / 'status')
                        processes.append({'pid':int(entry.name), 'rss_bytes': data.get('VmRSS', 0) * 1024})
                    except (FileNotFoundError, PermissionError, ProcessLookupError):
                        continue
            processes = sorted(processes, key=lambda p:p['rss_bytes'], reverse=True)[:8]
            snapshot = ResourceSnapshot(self.clock(), total, available, rss, sin, sout,
                paging_source='/proc/vmstat pswpin/pswpout * PAGE_SIZE',
                cgroup_limit_bytes=limit, cgroup_available_bytes=cgavailable,
                wsl=wsl, processes=processes, diagnostics=diagnostics)
            if wsl:
                host = _windows_host()
                snapshot.host_total_bytes = host['total_bytes']
                snapshot.host_available_bytes = host['available_bytes']
                snapshot.diagnostics['windows_host'] = host
                page_size = int(host['page_size'])
                frequency = float(host['raw_frequency'])
                if page_size <= 0 or frequency <= 0:
                    raise ValueError('Invalid Windows paging counter scale')
                snapshot.host_pagefile_in_bytes = int(host['hard_pages_input']) * page_size
                snapshot.host_pagefile_out_bytes = int(host['hard_pages_output']) * page_size
                snapshot.host_counter_time_seconds = float(host['raw_time']) / frequency
                snapshot.host_pagefile_verified = True
                snapshot.host_measurement_kind = 'upper_bound'
                snapshot.host_pagefile_source = 'Win32_PerfRawData_PerfOS_Memory PagesInputPersec/PagesOutputPersec cumulative counters'
                snapshot.diagnostics['host_pagefile_limitation'] = 'Disk hard-page I/O includes mapped files; pagefile-only I/O <= reported upper bound'
                adapters = host.get('gpu_adapters', [])
                if isinstance(adapters, dict):
                    adapters = [adapters]
                if adapters and all(a.get('SharedUsage') is not None and int(a['SharedUsage']) >= 0 for a in adapters):
                    snapshot.gpu_shared_used_bytes = sum(int(a['SharedUsage']) for a in adapters)
                    snapshot.gpu_shared_verified = True
                    snapshot.diagnostics['gpu_shared_scope'] = 'all Windows adapters; existing residency is included in host available RAM, growth is not attributed to this task'
        elif system == 'Darwin':
            total = int(_run(['sysctl', '-n', 'hw.memsize']))
            stat = _run(['vm_stat'])
            page = int(re.search(r'page size of (\d+) bytes', stat).group(1))
            values = {m[0]:int(m[1]) for m in re.findall(r'^([^:]+):\s+(\d+)\.', stat, re.M)}
            # Conservative reclaimable memory: do not count compressed pages or swap.
            available = (values['Pages free'] + values['Pages inactive'] + values['Pages speculative']) * page
            rss = int(_run(['ps', '-o', 'rss=', '-p', str(os.getpid())])) * 1024
            snapshot = ResourceSnapshot(self.clock(), total, available, rss,
                values['Swapins'] * page, values['Swapouts'] * page,
                paging_source='vm_stat Swapins/Swapouts * page_size')
            snapshot.diagnostics['available_definition'] = 'free + inactive + speculative; excludes compressed memory'
            snapshot.diagnostics['compressed_pages'] = values.get('Pages occupied by compressor')
            rows = _run(['ps', '-axo', 'pid=,rss=']).splitlines()
            snapshot.processes = sorted(
                ({'pid':int(parts[0]), 'rss_bytes':int(parts[1])*1024}
                 for line in rows if len(parts := line.split()) == 2),
                key=lambda item:item['rss_bytes'], reverse=True)[:8]
        else:
            raise ValueError('Unsupported native OS; supply a verified ResourceSnapshot probe')
        if self.device != 'cpu':
            (snapshot.gpu_total_bytes, snapshot.gpu_free_bytes,
             snapshot.gpu_process_bytes, snapshot.gpu_process_measurement_kind) = _gpu_probe(self.device)
            if snapshot.gpu_process_measurement_kind == 'exact':
                snapshot.gpu_process_source = 'nvidia-smi selected-device compute-apps used_memory for all matching current-PID rows'
            else:
                snapshot.gpu_process_source = 'nvidia-smi selected-device memory.total - memory.free'
                snapshot.diagnostics['gpu_process_limitation'] = (
                    'Current-PID dedicated residency is unavailable; whole-device used memory is an upper bound '
                    'including contexts and other processes. It is not attributed to this task or credited against remaining demand.')
            if snapshot.wsl and not snapshot.gpu_shared_verified:
                snapshot.diagnostics['gpu_shared_limitation'] = 'Windows GPUAdapterMemory counters unavailable; nvidia-smi is not a substitute'
        return snapshot


class ResourceGuard:
    def __init__(self, ram_budget_bytes, gpu_budget_bytes=0, device='cpu', log_path=None,
                 reserve_gib=8, reserve_fraction=.2, interval_seconds=5,
                 paging_grace_samples=2, probe=None, sample_seconds=1.0,
                 paging_threshold_bytes_per_second=MIB, clock=time.monotonic,
                 sleeper=time.sleep, max_sample_gap_seconds=300):
        self.ram_budget_bytes = int(ram_budget_bytes)
        self.gpu_budget_bytes = int(gpu_budget_bytes)
        self.device = str(device)
        self.reserve_gib, self.reserve_fraction = float(reserve_gib), float(reserve_fraction)
        self.interval_seconds, self.sample_seconds = float(interval_seconds), float(sample_seconds)
        self.paging_grace_samples = int(paging_grace_samples)
        self.paging_threshold = float(paging_threshold_bytes_per_second)
        self.max_sample_gap = float(max_sample_gap_seconds)
        for value in (self.reserve_gib, self.reserve_fraction, self.interval_seconds,
                      self.sample_seconds, self.paging_threshold, self.max_sample_gap):
            if not math.isfinite(value):
                raise ValueError('Resource parameters must be finite')
        if self.ram_budget_bytes <= 0 or self.gpu_budget_bytes < 0:
            raise ValueError('RAM peak budget must be positive and GPU peak budget nonnegative')
        if self.reserve_gib < 0 or not 0 <= self.reserve_fraction < 1:
            raise ValueError('Invalid RAM reserve; small environments require an explicit adapted reserve')
        if self.interval_seconds <= 0 or self.sample_seconds <= 0 or self.max_sample_gap <= 0:
            raise ValueError('Resource sampling intervals must be positive')
        if self.paging_grace_samples < 2 or self.paging_threshold <= 0:
            raise ValueError('Paging detection requires at least two valid windows and a positive threshold')
        if self.device == 'cpu' and self.gpu_budget_bytes:
            raise ValueError('CPU tasks cannot have a GPU budget')
        if self.device != 'cpu' and not self.gpu_budget_bytes:
            raise ValueError('GPU tasks require an explicit positive GPU peak budget')
        self.clock, self.sleeper = clock, sleeper
        self.probe = probe or SystemProbe(self.device, clock)
        self.log_path = Path(log_path) if log_path else None
        self._previous = None
        self._last_check = None
        self._paging_streak = {'linux_or_local':0, 'windows_host':0}
        self._admitted = False
        self._gpu_shared_baseline = None
        self._stopped = False
        self.peak_rss_bytes = 0
        self.peak_gpu_bytes = 0
        self.peak_gpu_exact_bytes = None
        self.peak_gpu_device_used_upper_bound_bytes = None

    def _receipt(self, status, evidence):
        includes_upper_bound = self.peak_gpu_device_used_upper_bound_bytes is not None
        gpu_peak_kind = ('device_used_upper_bound' if includes_upper_bound else
                         'exact' if self.peak_gpu_exact_bytes is not None else 'not_measured')
        gpu_peak_scope = ('selected device; includes contexts and other processes; not attributable to this task'
                          if includes_upper_bound else 'current process' if self.peak_gpu_exact_bytes is not None else 'not measured')
        event = {'status':status, 'reason':'resource_pressure' if status == 'stopped' else None,
                 'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 'pid':os.getpid(), 'device':self.device,
                 'ram_peak_budget_bytes':self.ram_budget_bytes,
                 'gpu_peak_budget_bytes':self.gpu_budget_bytes,
                 'observed_peak_rss_bytes':self.peak_rss_bytes,
                 'observed_peak_gpu_bytes':self.peak_gpu_bytes,
                 'observed_peak_gpu_measurement_kind':gpu_peak_kind,
                 'observed_peak_gpu_scope':gpu_peak_scope,
                 'observed_peak_gpu_exact_process_bytes':self.peak_gpu_exact_bytes,
                 'observed_peak_gpu_device_used_upper_bound_bytes':self.peak_gpu_device_used_upper_bound_bytes,
                 'resume_policy':'explicit_resume_after_resource_revalidation', **evidence}
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        return event

    def _stop(self, code, message, evidence=None):
        self._stopped = True
        data = {'code':code, 'message':message, **(evidence or {})}
        try:
            self._receipt('stopped', data)
        except (OSError, ValueError) as exc:
            data['receipt_error'] = type(exc).__name__ + ': ' + str(exc)
        raise ResourcePressureError(message, data)

    def _sample(self):
        try:
            snapshot = self.probe()
            if isinstance(snapshot, dict):
                snapshot = ResourceSnapshot(**snapshot)
            if not isinstance(snapshot, ResourceSnapshot):
                raise ValueError('Probe must return ResourceSnapshot or a matching mapping')
            if not isinstance(snapshot.timestamp, (int, float)) or not math.isfinite(snapshot.timestamp):
                raise ValueError('Invalid monotonic sample timestamp')
            values = [snapshot.ram_total_bytes, snapshot.ram_available_bytes,
                      snapshot.process_rss_bytes, snapshot.swap_in_bytes, snapshot.swap_out_bytes]
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('Missing or invalid RAM/RSS/actual swap counter')
            if snapshot.ram_total_bytes <= 0 or snapshot.ram_available_bytes > snapshot.ram_total_bytes:
                raise ValueError('Inconsistent physical memory counters')
            optional_numbers = (snapshot.cgroup_limit_bytes, snapshot.cgroup_available_bytes,
                snapshot.host_total_bytes, snapshot.host_available_bytes, snapshot.host_pagefile_in_bytes,
                snapshot.host_pagefile_out_bytes, snapshot.host_counter_time_seconds, snapshot.gpu_total_bytes,
                snapshot.gpu_free_bytes, snapshot.gpu_process_bytes, snapshot.gpu_shared_used_bytes)
            if any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0) for v in optional_numbers):
                raise ValueError('Invalid cgroup/host/GPU counter; missing is not zero')
            if snapshot.cgroup_available_bytes is not None and snapshot.cgroup_limit_bytes is None:
                raise ValueError('Cgroup availability lacks its effective limit')
            self.peak_rss_bytes = max(self.peak_rss_bytes, snapshot.process_rss_bytes)
            return snapshot
        except ResourcePressureError:
            raise
        except Exception as exc:
            self._stop('probe_failed', 'Resource state cannot be verified; do not start/continue new allocations.',
                       {'probe_error':type(exc).__name__ + ': ' + str(exc)})

    def _capacity(self, snapshot):
        evidence = {'snapshot':dataclasses.asdict(snapshot)}
        total, available = snapshot.ram_total_bytes, snapshot.ram_available_bytes
        if snapshot.cgroup_limit_bytes is not None:
            if snapshot.cgroup_available_bytes is None or not math.isfinite(snapshot.cgroup_limit_bytes) or not math.isfinite(snapshot.cgroup_available_bytes) or snapshot.cgroup_limit_bytes <= 0 or not 0 <= snapshot.cgroup_available_bytes <= snapshot.cgroup_limit_bytes:
                self._stop('cgroup_unverified', 'Effective cgroup RAM boundary is invalid.', evidence)
            total = min(total, snapshot.cgroup_limit_bytes)
            available = min(available, snapshot.cgroup_available_bytes)
        reserve = max(self.reserve_gib * GIB, self.reserve_fraction * total)
        remaining = max(0, self.ram_budget_bytes - snapshot.process_rss_bytes)
        evidence.update(effective_total_bytes=total, effective_available_bytes=available,
                        reserve_bytes=reserve, remaining_ram_budget_bytes=remaining)
        if snapshot.process_rss_bytes > self.ram_budget_bytes:
            self._stop('ram_budget_exceeded', 'Observed process RSS exceeds its registered peak budget; save state and revise the measured budget.', evidence)
        if available < reserve + remaining:
            self._stop('insufficient_ram', 'Available effective RAM cannot cover the remaining peak plus the configured reserve; reduce bounded work or wait.', evidence)
        if snapshot.wsl:
            if any(v is None or not math.isfinite(v) for v in (snapshot.host_total_bytes, snapshot.host_available_bytes)) or snapshot.host_total_bytes <= 0 or not 0 <= snapshot.host_available_bytes <= snapshot.host_total_bytes:
                self._stop('host_ram_unverified', 'WSL Windows host physical RAM could not be verified.', evidence)
            host_reserve = max(self.reserve_gib * GIB, self.reserve_fraction * snapshot.host_total_bytes)
            evidence['host_reserve_bytes'] = host_reserve
            if snapshot.host_available_bytes < host_reserve + remaining:
                self._stop('insufficient_host_ram', 'Windows host RAM cannot cover the WSL task remaining peak plus reserve.', evidence)
            if not snapshot.host_pagefile_verified or not snapshot.host_pagefile_source or snapshot.host_measurement_kind not in ('exact', 'upper_bound') or any(v is None or not math.isfinite(v) or v < 0 for v in (snapshot.host_pagefile_in_bytes, snapshot.host_pagefile_out_bytes, snapshot.host_counter_time_seconds)):
                self._stop('pagefile_actual_io_unverified', 'Windows pagefile I/O has neither verified exact counters nor a verified upper bound; provide a valid counter probe before running.', evidence)
        if self.device != 'cpu':
            vals = [snapshot.gpu_total_bytes, snapshot.gpu_free_bytes, snapshot.gpu_process_bytes]
            kind = snapshot.gpu_process_measurement_kind
            if (any(v is None or not math.isfinite(v) or v < 0 for v in vals)
                    or snapshot.gpu_total_bytes <= 0 or snapshot.gpu_free_bytes > snapshot.gpu_total_bytes
                    or snapshot.gpu_process_bytes > snapshot.gpu_total_bytes
                    or kind not in ('exact', 'device_used_upper_bound')
                    or not isinstance(snapshot.gpu_process_source, str) or not snapshot.gpu_process_source.strip()):
                self._stop('gpu_unverified', 'Dedicated GPU memory could not be verified.', evidence)
            if kind == 'device_used_upper_bound' and snapshot.gpu_process_bytes != snapshot.gpu_total_bytes - snapshot.gpu_free_bytes:
                self._stop('gpu_unverified', 'Device-used GPU upper bound does not match verified total minus free memory.', evidence)
            self.peak_gpu_bytes = max(self.peak_gpu_bytes, snapshot.gpu_process_bytes)
            if kind == 'exact':
                self.peak_gpu_exact_bytes = max(self.peak_gpu_exact_bytes or 0, snapshot.gpu_process_bytes)
                gpu_remaining = max(0, self.gpu_budget_bytes - snapshot.gpu_process_bytes)
                gpu_scope = 'current process'
            else:
                self.peak_gpu_device_used_upper_bound_bytes = max(
                    self.peak_gpu_device_used_upper_bound_bytes or 0, snapshot.gpu_process_bytes)
                gpu_remaining = self.gpu_budget_bytes
                gpu_scope = 'selected device; includes contexts and other processes; not attributable to this task'
            evidence.update(remaining_gpu_budget_bytes=gpu_remaining,
                            gpu_process_measurement_kind=kind, gpu_process_measurement_scope=gpu_scope)
            if snapshot.gpu_process_bytes > self.gpu_budget_bytes:
                if kind == 'device_used_upper_bound':
                    self._stop('gpu_budget_upper_bound_exceeded',
                               'Whole-device dedicated GPU usage exceeds the task budget, so its own peak cannot be certified. '
                               'This upper bound includes contexts and other processes; it does not establish that this task exceeded its budget. '
                               'Preserve progress and verify process-level residency or wait for lower device occupancy.', evidence)
                self._stop('gpu_budget_exceeded', 'Current-process dedicated GPU residency exceeds its peak budget.', evidence)
            if snapshot.gpu_free_bytes < gpu_remaining:
                self._stop('insufficient_gpu', 'Dedicated GPU free memory cannot cover the remaining peak; shared system memory is not GPU capacity.', evidence)
            if snapshot.wsl and (not snapshot.gpu_shared_verified or snapshot.gpu_shared_used_bytes is None or not math.isfinite(snapshot.gpu_shared_used_bytes) or snapshot.gpu_shared_used_bytes < 0):
                self._stop('gpu_shared_unverified', 'Windows shared GPU memory risk cannot be verified.', evidence)
            if snapshot.wsl:
                if self._gpu_shared_baseline is None:
                    self._gpu_shared_baseline = snapshot.gpu_shared_used_bytes
                if snapshot.gpu_shared_used_bytes - self._gpu_shared_baseline >= 64 * MIB:
                    self._stop('gpu_shared_growth', 'Windows shared GPU RAM increased by at least 64 MiB from admission baseline; preserve progress and verify the dedicated-memory budget. Other processes may also cause this growth.', evidence)
                evidence['gpu_shared_baseline_bytes'] = self._gpu_shared_baseline
        return evidence

    def _paging(self, previous, current, allow_deferred=False):
        rates = {}
        domains = [('linux_or_local',previous.timestamp,current.timestamp,
                    previous.swap_in_bytes,current.swap_in_bytes,previous.swap_out_bytes,current.swap_out_bytes)]
        if current.wsl != previous.wsl or current.paging_source != previous.paging_source:
            self._stop('counter_source_changed', 'Paging counter source changed; fresh admission is required.')
        if current.wsl:
            if current.host_pagefile_source != previous.host_pagefile_source or current.host_measurement_kind != previous.host_measurement_kind:
                self._stop('counter_source_changed', 'Windows pagefile counter source changed.')
            domains.append(('windows_host', previous.host_counter_time_seconds,current.host_counter_time_seconds,
                previous.host_pagefile_in_bytes,current.host_pagefile_in_bytes,
                previous.host_pagefile_out_bytes,current.host_pagefile_out_bytes))
        windows = []
        for name, t0, t1, i0, i1, o0, o1 in domains:
            values = [t0,t1,i0,i1,o0,o1]
            if any(v is None or not math.isfinite(v) for v in values):
                self._stop('invalid_paging_window', 'Paging window has missing or nonfinite counters.', {'domain':name})
            dt = t1 - t0
            # A forced capacity check can read the same Windows provider sample.
            # An unchanged timestamp is safe to defer only with identical
            # counters; it is never evidence that the paging rate was zero.
            same_host_sample = (allow_deferred and name == 'windows_host'
                                and dt == 0 and i1 == i0 and o1 == o0)
            if (not (0 < dt <= self.max_sample_gap or same_host_sample)
                    or i1 < i0 or o1 < o0):
                self._stop('invalid_paging_window', 'Paging counters reset or sample time is invalid; unknown is not zero paging.',
                           {'domain':name, 'elapsed_seconds':dt, 'input_delta':i1-i0, 'output_delta':o1-o0})
            windows.append((name, dt, i1-i0, o1-o0))
        if allow_deferred and any(dt < self.sample_seconds for _, dt, _, _ in windows):
            return {'measurement_status':'deferred',
                    'reason':'minimum_paging_window_not_reached',
                    'minimum_window_seconds':self.sample_seconds,
                    'elapsed_seconds_by_domain':{name:dt for name, dt, _, _ in windows},
                    'retained_consecutive_active_windows':dict(self._paging_streak)}
        for name, dt, incoming_delta, outgoing_delta in windows:
            incoming, outgoing = incoming_delta/dt, outgoing_delta/dt
            active = incoming + outgoing >= self.paging_threshold
            self._paging_streak[name] = self._paging_streak[name] + 1 if active else 0
            kind = current.host_measurement_kind if name == 'windows_host' else 'exact'
            rates[name] = {'measurement_kind':kind, 'elapsed_seconds':dt, 'in_bytes_per_second':incoming,
                           'out_bytes_per_second':outgoing, 'consecutive_active_windows':self._paging_streak[name]}
            if self._paging_streak[name] >= self.paging_grace_samples:
                code = 'host_paging_unresolved' if kind == 'upper_bound' else 'sustained_paging'
                message = ('Windows disk-paging upper bound is persistently high; actual pagefile I/O cannot be isolated. Preserve progress and resolve host paging before explicit resume.' if kind == 'upper_bound' else 'Sustained measured paging exceeds the configured threshold; preserve progress and explicitly revalidate before resuming.')
                self._stop(code, message,
                           {'paging':rates, 'threshold_bytes_per_second':self.paging_threshold,
                            'snapshot':dataclasses.asdict(current),
                            'attribution':'system paging observed; this does not establish which process caused it'})
        return rates

    def preflight(self):
        if self._stopped:
            self._stop('guard_already_stopped', 'This guard has stopped; an explicit new admission is required.')
        previous = self._sample()
        evidence = self._capacity(previous)
        # Two valid windows are required: a single burst is not sustained paging.
        for _ in range(self.paging_grace_samples):
            self.sleeper(self.sample_seconds)
            current = self._sample()
            evidence = self._capacity(current)
            evidence['paging'] = self._paging(previous, current)
            previous = current
        self._previous = previous
        self._last_check = self.clock()
        self._admitted = True
        try:
            return self._receipt('admitted', evidence)
        except (OSError, ValueError) as exc:
            self._stop('receipt_failed', 'Cannot persist resource admission evidence.', {'receipt_error':str(exc)})

    def check(self, force=False):
        if self._stopped:
            self._stop('guard_already_stopped', 'This guard has stopped; automatic continuation is forbidden.')
        if not self._admitted:
            self._stop('not_admitted', 'Call preflight() before loading the model or allocating task data.')
        now = self.clock()
        if not force and now - self._last_check < self.interval_seconds:
            return None
        current = self._sample()
        evidence = self._capacity(current)
        evidence['paging'] = self._paging(self._previous, current, allow_deferred=True)
        if evidence['paging'].get('measurement_status') != 'deferred':
            self._previous = current
        self._last_check = now
        try:
            return self._receipt('checked', evidence)
        except (OSError, ValueError) as exc:
            self._stop('receipt_failed', 'Cannot persist resource monitoring evidence.', {'receipt_error':str(exc)})


def add_resource_arguments(parser):
    group = parser.add_argument_group('Resource admission (peak budgets; no swap capacity)')
    group.add_argument('--ram-budget-gib', type=float, default=None,
                       help='Explicit total task peak RAM budget; default is the entry point estimate')
    group.add_argument('--gpu-budget-gib', type=float, default=None,
                       help='Explicit total task dedicated GPU peak budget')
    group.add_argument('--resource-reserve-gib', type=float, default=8,
                       help='Minimum physical RAM reserve; adapt explicitly for smaller hosts')
    group.add_argument('--resource-reserve-fraction', type=float, default=.2,
                       help='Fraction of effective physical RAM reserved (default 0.2)')
    group.add_argument('--resource-check-interval', type=float, default=5)
    group.add_argument('--resource-paging-threshold-mib-s', type=float, default=1)
    group.add_argument('--resource-paging-grace-samples', type=int, default=2)
    return parser


def guard_from_args(args, ram_budget_bytes, gpu_budget_bytes=0, device='cpu', log_path=None):
    def budget(name, fallback):
        value = getattr(args, name, None)
        if value is None:
            return fallback
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
        return int(value * GIB)
    return ResourceGuard(
        budget('ram_budget_gib', ram_budget_bytes), budget('gpu_budget_gib', gpu_budget_bytes),
        device=device, log_path=log_path,
        reserve_gib=getattr(args,'resource_reserve_gib',8),
        reserve_fraction=getattr(args,'resource_reserve_fraction',.2),
        interval_seconds=getattr(args,'resource_check_interval',5),
        paging_grace_samples=getattr(args,'resource_paging_grace_samples',2),
        paging_threshold_bytes_per_second=getattr(args,'resource_paging_threshold_mib_s',1)*MIB)
