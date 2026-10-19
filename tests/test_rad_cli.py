"""Mock-only CLI admission/recovery tests; no model or resource probe runs."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from torch.utils.data import Dataset

import rad_cli
from rad_bank import BankError
from rad_evaluation import atomic_json, prepare_signature


VALUES = [.75, .8, .7, .9, .85, .8, .65]


class TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        raise AssertionError('CLI orchestration tests must never decode an image')


class CliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(
            batch_size=2, num_workers=0, metric_chunk_size=11, resize_mask=8,
            pos_radius=1, vis_max=0, max_ratio=.25, reference_chunk_size=2,
            query_patch_chunk_size=2, cls_chunk_size=2, memory_workspace_mib=8,
            max_score_mib=8, save_dir=str(self.root), save_name='run', resume_eval=False,
            item_list=['sample'], bank_path=str(self.root / 'bank'), layer_weights=[1., 3.],
            encoder_name='dinov3_vitb16', encoder_weight=str(self.root / 'weights.pth'),
            data_path=str(self.root / 'data'), k_image=2, use_positional_bank=False,
            cleanup_metric_spool=False, ram_budget_gib=None, gpu_budget_gib=None,
        )
        self.bank = mock.Mock()
        self.bank.layers = [3, 6]
        self.bank.n_samples = 3
        self.bank.patch_shapes = {3: (3, 4, 4), 6: (3, 4, 4)}
        self.bank.provenance = {'categories': ['sample'], 'max_source_image_pixels': 1234567}
        self.guard = mock.Mock()
        self.encoder = mock.Mock()
        self.encoder.to.return_value = self.encoder
        self.encoder.eval.return_value = self.encoder
        self.encoder.parameters.return_value = ()
        stack = ExitStack()
        self.addCleanup(stack.close)
        def patch(name, **kwargs):
            return stack.enter_context(mock.patch.object(rad_cli, name, **kwargs))
        self.reader = patch('BankReader', return_value=self.bank)
        self.guard_factory = patch('guard_from_args', return_value=self.guard)
        self.load_model = patch('load_dinov3_model', return_value=self.encoder)
        self.estimate = patch('estimate_build_resources', side_effect=lambda **kwargs: {
            'ram_budget_bytes': 1024, 'gpu_budget_bytes': 0,
            'basis': 'synthetic estimate', 'components': {},
        })
        self.fingerprint = patch('checkpoint_fingerprint', side_effect=lambda path: f'sha:{Path(path).name}')
        self.transforms = patch('get_data_transforms', return_value=(object(), object()))
        self.dataset = patch('MVTecDataset', return_value=TinyDataset())
        self.realiad = patch('RealIADDataset', return_value=TinyDataset())
        self.signature = patch('dataset_signature', return_value={
            'sha256': 'synthetic-dataset', 'count': 2, 'max_source_pixels': 16,
        })
        self.evaluate = patch('eval_patch_knn_multilayer_cluster', side_effect=self.fake_evaluation)
        self.cuda = stack.enter_context(mock.patch.object(rad_cli.torch.cuda, 'is_available', return_value=False))

    @property
    def metric_dir(self):
        return self.root / self.args.save_name / 'metrics' / 'sample'

    def fake_evaluation(self, encoder, bank, loader, device, *, metric_dir, signature, **kwargs):
        # Publish only the provenance and completed-prediction receipt needed by
        # CLI recovery. The metric implementation has separate numerical tests.
        self.assertEqual(len(loader.dataset), 2)
        prepare_signature(metric_dir, signature, resume=kwargs['resume'])
        atomic_json(Path(metric_dir) / 'spool' / 'metric-state.json',
                    {'num_samples': 2, 'appended': 2})
        return list(VALUES)

    def run_cli(self):
        return rad_cli.run_evaluation(self.args, print_fn=lambda message: None)

    def test_bad_bank_provenance_is_rejected_before_model_or_dataset(self):
        self.bank.validate_provenance.side_effect = BankError('Mismatched checkpoint provenance')
        with self.assertRaisesRegex(BankError, 'Mismatched checkpoint'):
            self.run_cli()
        self.load_model.assert_not_called()
        self.dataset.assert_not_called()
        self.evaluate.assert_not_called()

    def test_existing_output_without_resume_never_loads_model(self):
        for existing_name in ('evaluation-inputs.json', 'unrelated.txt'):
            with self.subTest(existing_name=existing_name):
                self.args.save_name = existing_name.replace('.', '-')
                self.metric_dir.mkdir(parents=True)
                (self.metric_dir / existing_name).write_text('{}')
                with self.assertRaisesRegex(FileExistsError, 'Existing evaluation'):
                    self.run_cli()
                self.load_model.assert_not_called()
                self.evaluate.assert_not_called()

    def test_changed_resume_signature_never_loads_model(self):
        self.metric_dir.mkdir(parents=True)
        (self.metric_dir / 'evaluation-inputs.json').write_text('{"old": "protocol"}')
        self.args.resume_eval = True
        with self.assertRaisesRegex(ValueError, 'protocol'):
            self.run_cli()
        self.load_model.assert_not_called()
        self.evaluate.assert_not_called()

    def test_later_existing_category_is_rejected_before_any_model_load(self):
        self.args.item_list = ['new', 'existing']
        self.bank.provenance['categories'] = ['new', 'existing']
        existing = self.root / self.args.save_name / 'metrics' / 'existing'
        existing.mkdir(parents=True)
        (existing / 'evaluation-inputs.json').write_text('{}')
        with self.assertRaisesRegex(FileExistsError, 'Existing evaluation'):
            self.run_cli()
        self.load_model.assert_not_called()
        self.dataset.assert_not_called()
        self.evaluate.assert_not_called()

    def test_complete_predictions_resume_uses_cpu_without_cuda_or_encoder(self):
        for completed_result in (False, True):
            with self.subTest(completed_result=completed_result):
                self.args.save_name = f'resume-{completed_result}'
                self.args.resume_eval = False
                self.args.gpu_budget_gib = None
                self.run_cli()  # all heavyweight operations above are mocks
                if completed_result:
                    atomic_json(self.metric_dir / 'metrics.json', {'values': VALUES})
                self.args.resume_eval = True
                self.args.gpu_budget_gib = 2.5  # a replayed inference command
                self.args.ram_budget_gib = 1.25
                self.load_model.reset_mock()
                self.estimate.reset_mock()
                self.evaluate.reset_mock()
                self.guard_factory.reset_mock()
                with mock.patch.object(rad_cli.torch.cuda, 'is_available',
                                       side_effect=AssertionError('CPU recovery queried CUDA')):
                    actual = self.run_cli()
                self.assertEqual(actual, VALUES)
                self.load_model.assert_not_called()
                self.estimate.assert_not_called()  # no encoder/forward resource estimator
                guard_call = self.guard_factory.call_args.args
                self.assertIsNone(guard_call[0].gpu_budget_gib)
                self.assertEqual(guard_call[0].ram_budget_gib, 1.25)
                self.assertEqual(self.args.gpu_budget_gib, 2.5)  # original args remain auditable
                self.assertEqual(guard_call[2], 0)
                self.assertEqual(guard_call[3], 'cpu')
                execution = json.loads((self.root / self.args.save_name / 'sample-execution.json').read_text())
                self.assertTrue(execution['metrics_only'])
                self.assertEqual(execution['gpu_budget_ignored_for_cpu_metrics'], 2.5)
                if completed_result:
                    self.evaluate.assert_not_called()
                else:
                    self.evaluate.assert_called_once()
                    self.assertIsNone(self.evaluate.call_args.args[0])
                    self.assertEqual(self.evaluate.call_args.args[3], 'cpu')
                    self.assertTrue(self.evaluate.call_args.kwargs['resume'])

    def test_visualization_estimate_includes_reference_source_image_size(self):
        for vis_max in (0, 1):
            with self.subTest(vis_max=vis_max):
                self.args.save_name = f'visualization-{vis_max}'
                self.args.vis_max = vis_max
                self.estimate.reset_mock()
                self.run_cli()
                values = [call.kwargs['max_source_pixels'] for call in self.estimate.call_args_list]
                self.assertTrue(values)
                self.assertEqual(values[-1], 1234567 if vis_max else 16)
                self.assertTrue(all(value == 1234567 for value in values) if vis_max
                                else all(value <= 16 for value in values))

    def test_cached_results_honor_explicit_later_cleanup_without_encoder(self):
        self.run_cli()
        atomic_json(self.metric_dir / 'metrics.json', {'values': VALUES})
        self.args.resume_eval = True
        self.args.cleanup_metric_spool = True
        self.load_model.reset_mock()
        self.evaluate.reset_mock()
        completed = mock.MagicMock()
        completed.__enter__.return_value = completed
        with mock.patch('rad_evaluation.MetricAccumulator.resume', return_value=completed) as reopen:
            with mock.patch.object(rad_cli.torch.cuda, 'is_available',
                                   side_effect=AssertionError('Cached cleanup queried CUDA')):
                self.assertEqual(self.run_cli(), VALUES)
        reopen.assert_called_once_with(self.metric_dir / 'spool', guard=self.guard)
        completed.cleanup.assert_called_once()
        self.load_model.assert_not_called()
        self.evaluate.assert_not_called()
        self.assertTrue((self.metric_dir / 'metrics.json').exists())

    def test_large_output_resolution_budgets_the_full_prediction_batch(self):
        self.args.batch_size = 32
        self.args.resize_mask = 4096
        result = rad_cli._budget(self.args, self.bank, 'cpu')
        # Merely materializing map + float32 sort values + int64 indices needs
        # 16 bytes/pixel for every query; convolution/GT/copies need more.
        unavoidable_sort_peak = 32 * 4096 ** 2 * 16
        self.assertGreater(result['ram_budget_bytes'], unavoidable_sort_peak)
        self.assertGreater(result['components']['prediction_workspace_bytes'], unavoidable_sort_peak)
        resumed = rad_cli._budget(self.args, self.bank, 'cpu', metrics_only=True)
        self.assertEqual(resumed['components']['prediction_workspace_bytes'], 0)
        self.assertGreaterEqual(resumed['components']['aupro_image_workspace_bytes'], 4096 ** 2 * 1024)
        self.assertIn('not an OS memory limit', resumed['basis'])


if __name__ == '__main__':
    unittest.main()
