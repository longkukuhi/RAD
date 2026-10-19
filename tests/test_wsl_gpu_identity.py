"""No GPU or Windows needed: identity APIs and host reads are simulated."""
import copy
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rad_resources import (
    GIB, MIB, SystemProbe, _cuda_device_identity, _map_windows_gpu_identity,
    _selected_windows_gpu_counter, _windows_gpu_identities,
)


UUID = 'GPU-12345678-1234-1234-1234-123456789abc'
OTHER_UUID = 'GPU-22345678-1234-1234-1234-123456789abc'
LUID = 'luid_0x00000000_0x00000123'
OTHER_LUID = 'luid_0x00000000_0x00000456'
SELECTED = {'uuid':UUID, 'cuda_ordinal':0, 'source':'injected CUDA Driver identity'}


def inventory():
    return {'source':'injected Windows native CUDA Driver', 'devices':[
        {'uuid':UUID, 'luid':LUID, 'node_mask':1},
        {'uuid':OTHER_UUID, 'luid':OTHER_LUID, 'node_mask':1},
    ]}


def host():
    return dict(total_bytes=64*GIB, available_bytes=48*GIB,
        hard_pages_input=0, hard_pages_output=0, raw_time=100, raw_frequency=10,
        page_size=4096, gpu_adapters=[
            {'Name':LUID+'_phys_0', 'SharedUsage':3*MIB, 'DedicatedUsage':100*MIB},
            {'Name':OTHER_LUID+'_phys_0', 'SharedUsage':5*MIB, 'DedicatedUsage':200*MIB},
        ])


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
    def __call__(self, *arguments):
        return self.callback(*arguments)


def fake_driver(*, error=0, legacy=False):
    calls = []
    def record(name, function):
        def invoke(*arguments):
            calls.append((name, arguments))
            if error:
                return error
            function(*arguments)
            return 0
        return FakeFunction(invoke)
    raw_uuid = bytes.fromhex(UUID[4:].replace('-', ''))
    driver = SimpleNamespace(
        cuInit=record('cuInit', lambda flags:None),
        cuDeviceGetCount=record('cuDeviceGetCount', lambda out:setattr(out._obj, 'value', 2)),
        cuDeviceGet=record('cuDeviceGet', lambda out, ordinal:setattr(out._obj, 'value', ordinal+10)),
    )
    def assign_uuid(out, handle):
        out[:] = raw_uuid
    name = 'cuDeviceGetUuid' if legacy else 'cuDeviceGetUuid_v2'
    setattr(driver, name, record(name, assign_uuid))
    return driver, calls


class IdentityTests(unittest.TestCase):
    def test_actual_cuda_ordinal_is_resolved_by_driver_without_context(self):
        driver, calls = fake_driver()
        with patch('rad_resources.ctypes.CDLL', return_value=driver) as load,\
                patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES':'2,1', 'CUDA_DEVICE_ORDER':'PCI_BUS_ID'}):
            result = _cuda_device_identity('cuda:1')
        self.assertEqual(result['uuid'], UUID)
        self.assertEqual(result['cuda_ordinal'], 1)
        self.assertEqual([name for name,_ in calls], ['cuInit','cuDeviceGetCount','cuDeviceGet','cuDeviceGetUuid_v2'])
        self.assertEqual(calls[2][1][1], 1)
        load.assert_called_once_with('libcuda.so.1')

    def test_cuda_legacy_uuid_api_is_supported(self):
        driver, calls = fake_driver(legacy=True)
        with patch('rad_resources.ctypes.CDLL', return_value=driver):
            self.assertEqual(_cuda_device_identity('cuda')['uuid'], UUID)
        self.assertEqual(calls[-1][0], 'cuDeviceGetUuid')

    def test_cuda_error_or_hidden_ordinal_fails_closed(self):
        driver, _ = fake_driver(error=100)
        with patch('rad_resources.ctypes.CDLL', return_value=driver), self.assertRaises(ValueError):
            _cuda_device_identity('cuda:0')
        driver, _ = fake_driver()
        with patch('rad_resources.ctypes.CDLL', return_value=driver), self.assertRaises(ValueError):
            _cuda_device_identity('cuda:2')

    def test_native_windows_mapping_uses_uuid_not_ordinal_or_name(self):
        rows = inventory()
        rows['devices'].reverse()
        result = _map_windows_gpu_identity(SELECTED, rows)
        self.assertEqual(result['luid'], LUID)
        self.assertEqual(result['node_mask'], 1)
        self.assertEqual(result['uuid'], UUID)

    def test_missing_or_duplicate_uuid_and_reused_luid_are_rejected(self):
        for mutation in ('missing', 'duplicate_uuid', 'duplicate_luid'):
            with self.subTest(mutation=mutation):
                rows = inventory()
                if mutation == 'missing':
                    rows['devices'].pop(0)
                elif mutation == 'duplicate_uuid':
                    rows['devices'][1]['uuid'] = UUID
                else:
                    rows['devices'][1]['luid'] = LUID
                with self.assertRaises(ValueError):
                    _map_windows_gpu_identity(SELECTED, rows)

    def test_invalid_mapping_metadata_is_not_a_zero_identity(self):
        for field, value in [('luid',None),('luid','luid_0x00000000_0x00000000'),
                             ('node_mask',0),('node_mask',None),('node_mask',True),
                             ('node_mask',2),('node_mask',3),('uuid','GPU-0')]:
            with self.subTest(field=field, value=value):
                rows=inventory();rows['devices'][0][field]=value
                with self.assertRaises(ValueError):
                    _map_windows_gpu_identity(SELECTED,rows)
        rows=inventory();rows['source']=''
        with self.assertRaises(ValueError):
            _map_windows_gpu_identity(SELECTED,rows)
        selected=dict(SELECTED);selected['source']=''
        with self.assertRaises(ValueError):
            _map_windows_gpu_identity(selected,inventory())

    def test_target_counter_ignores_unrelated_adapter_usage(self):
        mapping=_map_windows_gpu_identity(SELECTED,inventory())
        result=_selected_windows_gpu_counter(host(),mapping)
        self.assertEqual(result['SharedUsage'],3*MIB)
        self.assertEqual(result['Name'],LUID+'_phys_0')

    def test_any_valid_other_adapter_cannot_replace_missing_target(self):
        data=host();data['gpu_adapters'].pop(0)
        with self.assertRaisesRegex(ValueError,'missing is not zero'):
            _selected_windows_gpu_counter(data,_map_windows_gpu_identity(SELECTED,inventory()))

    def test_duplicate_target_rows_are_ambiguous(self):
        data=host();data['gpu_adapters'].append(dict(data['gpu_adapters'][0]))
        with self.assertRaises(ValueError):
            _selected_windows_gpu_counter(data,_map_windows_gpu_identity(SELECTED,inventory()))

    def test_unsupported_or_changed_physical_node_is_rejected(self):
        for name in (LUID, LUID+'_phys_1', LUID+'_phys_00'):
            with self.subTest(name=name):
                data=host();data['gpu_adapters'][0]['Name']=name
                with self.assertRaises(ValueError):
                    _selected_windows_gpu_counter(data,_map_windows_gpu_identity(SELECTED,inventory()))

    def test_missing_or_invalid_target_usage_fails_closed(self):
        for field in ('SharedUsage','DedicatedUsage'):
            for value in (None, 'N/A', float('nan'), -1, True, 0.5):
                with self.subTest(field=field,value=value):
                    data=host();data['gpu_adapters'][0][field]=value
                    with self.assertRaises(ValueError):
                        _selected_windows_gpu_counter(data,_map_windows_gpu_identity(SELECTED,inventory()))

    def test_windows_native_probe_invalid_source_is_rejected(self):
        with patch('rad_resources._run',return_value='{"devices":[]}'),self.assertRaises(ValueError):
            _windows_gpu_identities()


class SystemProbeTests(unittest.TestCase):
    def mocked_system(self, windows=True):
        stack=ExitStack()
        stack.enter_context(patch('rad_resources.platform.system',return_value='Linux'))
        def numbers(path):
            return {'/proc/meminfo':{'MemTotal':64*GIB//1024,'MemAvailable':48*GIB//1024},
                    '/proc/vmstat':{'pswpin':0,'pswpout':0},
                    '/proc/self/status':{'VmRSS':100*MIB//1024}}[str(path)]
        stack.enter_context(patch('rad_resources._read_numbers',side_effect=numbers))
        stack.enter_context(patch('rad_resources._linux_cgroups',return_value=(None,None,[])))
        stack.enter_context(patch('rad_resources.os.sysconf',return_value=4096))
        stack.enter_context(patch.object(Path,'read_text',return_value='microsoft-standard-WSL2' if windows else 'linux'))
        stack.enter_context(patch.object(Path,'exists',return_value=False))
        stack.enter_context(patch.object(Path,'iterdir',return_value=iter(())))
        return stack

    def test_wsl_identity_cached_but_current_target_counter_required_every_check(self):
        missing=host();missing['gpu_adapters'].pop(0)
        with self.mocked_system(),\
                patch('rad_resources._windows_host',side_effect=[host(),host(),missing]),\
                patch('rad_resources._cuda_device_identity',return_value=copy.deepcopy(SELECTED)) as cuda,\
                patch('rad_resources._windows_gpu_identities',return_value=inventory()) as native,\
                patch('rad_resources._gpu_probe',return_value=(32*GIB,30*GIB,2*GIB,'device_used_upper_bound')):
            probe=SystemProbe('cuda:0')
            for _ in range(2):
                result=probe()
                self.assertTrue(result.gpu_shared_verified)
                self.assertEqual(result.gpu_shared_used_bytes,3*MIB)
                self.assertEqual(result.diagnostics['gpu_identity']['luid'],LUID)
            with self.assertRaisesRegex(ValueError,'missing is not zero'):
                probe()
            cuda.assert_called_once_with('cuda:0')
            native.assert_called_once_with()

    def test_cpu_never_loads_cuda_driver_or_native_identity(self):
        for windows in (False,True):
            with self.subTest(windows=windows),self.mocked_system(windows),\
                    patch('rad_resources._windows_host',return_value=host()),\
                    patch('rad_resources.ctypes.CDLL',side_effect=AssertionError('CPU loaded CUDA')),\
                    patch('rad_resources._windows_gpu_identities',side_effect=AssertionError('CPU queried CUDA')):
                self.assertIsNone(SystemProbe('cpu')().gpu_process_bytes)

    def test_counter_identity_change_after_admission_is_rejected(self):
        changed=host();changed['gpu_adapters'][0]['Name']=LUID+'_phys_1'
        with self.mocked_system(),\
                patch('rad_resources._windows_host',side_effect=[host(),changed]),\
                patch('rad_resources._cuda_device_identity',return_value=copy.deepcopy(SELECTED)),\
                patch('rad_resources._windows_gpu_identities',return_value=inventory()),\
                patch('rad_resources._gpu_probe',return_value=(32*GIB,30*GIB,2*GIB,'device_used_upper_bound')):
            probe=SystemProbe('cuda:0')
            probe()
            with self.assertRaisesRegex(ValueError,'physical-node identity'):
                probe()

    def test_selection_environment_change_cannot_reuse_cached_identity(self):
        probe=SystemProbe('cuda:0')
        with patch('rad_resources._cuda_device_identity',return_value=copy.deepcopy(SELECTED)),\
                patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0'}):
            probe._selected_gpu()
            with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'1'}),self.assertRaises(ValueError):
                probe._selected_gpu()


if __name__ == '__main__':
    unittest.main()
