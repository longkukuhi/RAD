"""Tiny end-to-end checks; no real image files, models or accelerator required."""
import json
import math
from pathlib import Path
from statistics import mean
import tempfile
import unittest
from unittest import mock

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from skimage import measure
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from rad_bank import BankError, BankReader, BankWriter
from rad_evaluation import eval_patch_knn_multilayer_cluster
from rad_retrieval import RetrievalConfig


LAYERS = [3, 6]
PROVENANCE = {'encoder': {'name': 'synthetic', 'checkpoint_sha256': 'synthetic-v1'},
              'preprocessing': {'crop_size': 4}, 'categories': ['test']}


class TinyDataset(Dataset):
    def __init__(self):
        self.labels = torch.tensor([0, 1, 1, 0, 1, 1])
        self.masks = torch.zeros(6, 1, 4, 4)
        self.masks[1, :, :2, :2] = 1
        self.masks[2, :, 1:3, 1:3] = 1
        self.masks[4, :, 2:, 2:] = 1
        self.masks[5, :, 0, 0] = self.masks[5, :, 3, 3] = 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.full((3, 4, 4), float(index)), self.masks[index], self.labels[index], f'{index}.png'


class FakeEncoder(torch.nn.Module):
    def __init__(self, patches, cls):
        super().__init__()
        self.patches, self.cls = patches, cls
        self.seen = []

    def get_intermediate_layers(self, images, n, return_class_token, reshape, norm):
        if (list(n), return_class_token, reshape, norm) != (LAYERS, True, False, True):
            raise AssertionError('Encoder feature protocol changed')
        ids = images[:, 0, 0, 0].to(torch.long)
        self.seen.extend(ids.tolist())
        return [(self.patches[layer][ids], self.cls[layer][ids]) for layer in n]


def tiny_features():
    generator = torch.Generator().manual_seed(17)
    bank_cls = {layer: torch.randn(7, 4, generator=generator) for layer in LAYERS}
    bank_patches = {layer: torch.randn(7, 4, 4, generator=generator) for layer in LAYERS}
    query_cls = {layer: torch.randn(6, 4, generator=generator) for layer in LAYERS}
    query_patches = {layer: torch.randn(6, 4, 4, generator=generator) for layer in LAYERS}
    return bank_cls, bank_patches, query_cls, query_patches


def disk_bank(path, cls, patches):
    with BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE) as writer:
        for start, stop in ((0, 3), (3, 7)):
            writer.append({layer: cls[layer][start:stop] for layer in LAYERS},
                          {layer: patches[layer][start:stop] for layer in LAYERS})
        return writer.finalize()


def dense_reference(dataset, bank_cls, bank_patches, query_cls, query_patches,
                    positional=False, radius=1, ratio=.25):
    """Original dense retrieval + interpolation/Gaussian/pooling definitions."""
    ids = (F.normalize(query_cls[6], dim=-1) @ F.normalize(bank_cls[6], dim=-1).t()).topk(3, dim=-1).indices
    fused = []
    for b in range(6):
        layers = []
        for layer in LAYERS:
            query = F.normalize(query_patches[layer][b], dim=-1)
            neighbors = F.normalize(bank_patches[layer][ids[b]], dim=-1)
            if positional:
                scores = []
                for patch in range(4):
                    row, col = divmod(patch, 2)
                    positions = [r * 2 + c
                                 for r in range(max(0, row-radius), min(2, row+radius+1))
                                 for c in range(max(0, col-radius), min(2, col+radius+1))]
                    local = neighbors[:, positions].reshape(-1, 4)
                    scores.append(1 - (query[patch:patch+1] @ local.t()).max())
                layers.append(torch.stack(scores))
            else:
                layers.append(1 - (query @ neighbors.reshape(-1, 4).t()).max(dim=-1).values)
        fused.append(.25 * layers[0] + .75 * layers[1])
    maps = F.interpolate(torch.stack(fused).view(6, 1, 2, 2), size=8, mode='bilinear', align_corners=False)
    # Construct the historical kernel independently of rad_evaluation.
    coordinates = torch.arange(5)
    x = coordinates.repeat(5).view(5, 5)
    grid = torch.stack([x, x.t()], dim=-1).float()
    kernel = (1 / (2 * math.pi)) * torch.exp(-torch.sum((grid - 2.) ** 2., dim=-1) / 2)
    kernel = (kernel / kernel.sum()).view(1, 1, 5, 5)
    maps = F.conv2d(maps, kernel, padding=2)[:, 0]
    masks = F.interpolate(dataset.masks, size=8, mode='nearest')[:, 0].bool()
    flat = maps.flatten(1)
    scores = flat.max(dim=1).values if ratio == 0 else flat.sort(dim=1, descending=True).values[:, :int(64*ratio)].mean(dim=1)
    return masks.numpy(), maps.numpy(), dataset.labels.numpy(), scores.numpy()


def reference_metrics(masks, maps, labels, scores):
    def ranking(y, values):
        precision, recall, _ = precision_recall_curve(y, values)
        f1 = (2 * precision * recall / (precision + recall + 1e-7))[:-1].max()
        return [roc_auc_score(y, values), average_precision_score(y, values), f1]
    minimum, maximum = maps.min(), maps.max()
    records = []
    for threshold in np.arange(minimum, maximum, (maximum-minimum) / 200):
        binary = maps > threshold
        regions = []
        for binary_map, mask in zip(binary, masks):
            for region in measure.regionprops(measure.label(mask)):
                rows, columns = region.coords[:, 0], region.coords[:, 1]
                regions.append(binary_map[rows, columns].sum() / region.area)
        background = ~masks
        fpr = np.logical_and(background, binary).sum() / background.sum()
        if fpr < .3:
            records.append((mean(regions), fpr))
    retained = np.asarray(records)
    pro = auc(retained[:, 1] / retained[:, 1].max(), retained[:, 0])
    return [*ranking(labels, scores), *ranking(masks.ravel(), maps.ravel()), pro]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.dataset = TinyDataset()
        self.bank_cls, self.bank_patches, self.query_cls, self.query_patches = tiny_features()
        self.loader = DataLoader(self.dataset, batch_size=2, shuffle=False, num_workers=0)

    def encoder(self):
        return FakeEncoder(self.query_patches, self.query_cls)

    def evaluate(self, encoder, bank, path, **overrides):
        config = dict(metric_dir=path, signature={'test': 'v1', 'bank': PROVENANCE},
                      k_image=3, resize_mask=8, max_ratio=.25, layer_weights=[1., 3.],
                      retrieval_config=RetrievalConfig(2, 3, 3, 8, 8), metric_chunk_size=11)
        config.update(overrides)
        return eval_patch_knn_multilayer_cluster(encoder, bank, self.loader, 'cpu', **config)

    def expected(self, positional=False, ratio=.25):
        return reference_metrics(*dense_reference(self.dataset, self.bank_cls, self.bank_patches,
                                                  self.query_cls, self.query_patches, positional,
                                                  0 if positional else 1, ratio))

    def test_disk_bank_end_to_end_seven_metrics_match_dense(self):
        with tempfile.TemporaryDirectory() as folder:
            bank = disk_bank(Path(folder) / 'bank', self.bank_cls, self.bank_patches)
            for positional in (False, True):
                for ratio in (0, .25):
                    with self.subTest(positional=positional, ratio=ratio):
                        path = Path(folder) / f'eval-{positional}-{ratio}'
                        encoder = self.encoder()
                        actual = self.evaluate(encoder, bank, path, use_positional_bank=positional,
                                               pos_radius=0 if positional else 1, max_ratio=ratio)
                        np.testing.assert_allclose(actual, self.expected(positional, ratio), rtol=0, atol=1e-6)
                        self.assertEqual(encoder.seen, list(range(6)))
                        record = json.loads((path / 'metrics.json').read_text())
                        self.assertEqual(record['sample_count'], 6)
                        self.assertEqual(record['values'], actual)
                        self.assertLessEqual(record['retrieval_peak_reference_rows'], 2)

    def test_resume_skips_committed_encoder_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            bank = disk_bank(Path(folder) / 'bank', self.bank_cls, self.bank_patches)
            path = Path(folder) / 'eval'
            state_path = path / 'spool' / 'metric-state.json'
            class StopAfterFirstBatch:
                def check(self):
                    if state_path.exists() and json.loads(state_path.read_text())['appended'] >= 2:
                        raise RuntimeError('synthetic safe stop after committed batch')
            first = self.encoder()
            with self.assertRaisesRegex(RuntimeError, 'synthetic safe stop'):
                self.evaluate(first, bank, path, guard=StopAfterFirstBatch())
            self.assertEqual(first.seen, [0, 1])
            receipt = json.loads((path / 'interrupted.json').read_text())
            self.assertEqual(receipt['committed_samples'], 2)
            resumed = self.encoder()
            actual = self.evaluate(resumed, bank, path, resume=True)
            self.assertEqual(resumed.seen, [2, 3, 4, 5])
            np.testing.assert_allclose(actual, self.expected(), rtol=0, atol=1e-6)

    def test_complete_spool_resume_runs_only_metrics_and_cached_result_survives_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            bank = disk_bank(Path(folder) / 'bank', self.bank_cls, self.bank_patches)
            path = Path(folder) / 'eval'
            state_path = path / 'spool' / 'metric-state.json'
            class StopBeforeMetrics:
                def check(self):
                    if state_path.exists() and json.loads(state_path.read_text())['appended'] == 6:
                        raise RuntimeError('synthetic stop before sorting')
            first = self.encoder()
            with self.assertRaisesRegex(RuntimeError, 'synthetic stop before sorting'):
                self.evaluate(first, bank, path, guard=StopBeforeMetrics())
            self.assertEqual(first.seen, list(range(6)))
            with mock.patch.object(DataLoader, '__iter__',
                                   side_effect=AssertionError('Metrics-only recovery iterated a DataLoader')) as iteration:
                actual = self.evaluate(None, bank, path, resume=True, cleanup_metric_spool=True)
                iteration.assert_not_called()
            self.assertEqual({entry.name for entry in (path / 'spool').iterdir()}, {'.metric.lock'})
            cached = self.encoder()
            self.assertEqual(self.evaluate(cached, bank, path, resume=True), actual)
            self.assertEqual(cached.seen, [])
            np.testing.assert_allclose(actual, self.expected(), rtol=0, atol=1e-6)

    def test_changed_provenance_or_evaluation_signature_fails_before_encoder(self):
        with tempfile.TemporaryDirectory() as folder:
            bank = disk_bank(Path(folder) / 'bank', self.bank_cls, self.bank_patches)
            with self.assertRaisesRegex(BankError, 'Mismatched'):
                BankReader(bank.path, expected_provenance={'encoder': {'checkpoint_sha256': 'changed'}})
            path = Path(folder) / 'eval'
            self.evaluate(self.encoder(), bank, path)
            encoder = self.encoder()
            with self.assertRaisesRegex(ValueError, 'inputs/protocol changed'):
                self.evaluate(encoder, bank, path, signature={'test': 'changed'}, resume=True)
            self.assertEqual(encoder.seen, [])
            with self.assertRaisesRegex(FileExistsError, 'Existing evaluation'):
                self.evaluate(encoder, bank, path)
            self.assertEqual(encoder.seen, [])

    def test_cached_result_cleanup_preserves_published_results_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as folder:
            bank = disk_bank(Path(folder) / 'bank', self.bank_cls, self.bank_patches)
            path = Path(folder) / 'eval'
            expected = self.evaluate(self.encoder(), bank, path)
            spool = path / 'spool'
            self.assertTrue((spool / 'metric-state.json').exists())
            note = spool / 'user-note.txt'
            note.write_text('retain this unrelated file')
            with mock.patch.object(DataLoader, '__iter__',
                                   side_effect=AssertionError('Cached cleanup iterated a DataLoader')):
                actual = self.evaluate(None, bank, path, resume=True, cleanup_metric_spool=True)
                self.assertEqual(actual, expected)
                self.assertFalse((spool / 'metric-state.json').exists())
                self.assertFalse((spool / 'maps.bin').exists())
                self.assertFalse((spool / 'sort-pixel').exists())
                self.assertEqual(note.read_text(), 'retain this unrelated file')
                self.assertTrue((path / 'metrics.json').exists())
                self.assertTrue((path / 'evaluation-inputs.json').exists())
                self.assertEqual(self.evaluate(None, bank, path, resume=True, cleanup_metric_spool=True), expected)

    def test_shuffled_or_dropped_samples_fail_before_encoder(self):
        with tempfile.TemporaryDirectory() as folder:
            bank = disk_bank(Path(folder) / 'bank', self.bank_cls, self.bank_patches)
            for shuffle, drop_last in ((True, False), (False, True)):
                with self.subTest(shuffle=shuffle, drop_last=drop_last):
                    self.loader = DataLoader(self.dataset, batch_size=2, shuffle=shuffle, drop_last=drop_last)
                    encoder = self.encoder()
                    with self.assertRaisesRegex(ValueError, 'sequential DataLoader'):
                        self.evaluate(encoder, bank, Path(folder) / 'invalid')
                    self.assertEqual(encoder.seen, [])


if __name__ == '__main__':
    unittest.main()
