"""Small CPU regressions against the original metric definitions.

Run with ``python -m unittest discover -s tests -p test_rad_metrics.py``.
No model weights, training data, CUDA, or torch import is required.
"""

import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from skimage import measure

from rad_metrics import (MetricAccumulator, _comparison_thresholds,
                         _external_ranking_metrics, compute_pro)


def original_pro_with_compatible_apis(masks, amaps, num_th=200):
    """Original compute_pro arithmetic; replace only np.bool/pandas._append."""
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)
    minimum, maximum = amaps.min(), amaps.max()
    delta = (maximum - minimum) / num_th
    records = []
    for threshold in np.arange(minimum, maximum, delta):
        binary_amaps[amaps <= threshold] = 0
        binary_amaps[amaps > threshold] = 1
        pros = []
        for binary_map, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                rows, columns = region.coords[:, 0], region.coords[:, 1]
                pros.append(binary_map[rows, columns].sum() / region.area)
        inverse_masks = 1 - masks.astype(np.uint8)
        false_positive = np.logical_and(inverse_masks, binary_amaps).sum()
        records.append((mean(pros), false_positive / inverse_masks.sum()))
    retained = np.asarray([(pro, fpr) for pro, fpr in records if fpr < 0.3])
    return auc(retained[:, 1] / retained[:, 1].max(), retained[:, 0])


def sklearn_metrics(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = (2 * precision * recall / (precision + recall + 1e-7))[:-1].max()
    return roc_auc_score(labels, scores), average_precision_score(labels, scores), f1


def sample_data(dtype=np.float32):
    rng = np.random.default_rng(312)
    masks = np.zeros((6, 7, 9), dtype=np.bool_)
    masks[1, 1:4, 1:3] = 1
    masks[2, 1, 1] = masks[2, 2, 2] = 1  # default connectivity joins diagonals
    masks[3, 2:5, 3:7] = 1
    masks[5, 1:3, 1:3] = masks[5, 5:7, 7:9] = 1
    maps = (rng.integers(0, 40, size=masks.shape) / 39).astype(dtype)
    labels = np.array([0, 1, 1, 1, 0, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.7, 0.7, 0.9, 0.4, 0.7], dtype=dtype)
    return masks, maps, labels, scores


class MetricDefinitionTests(unittest.TestCase):
    def test_aupro_matches_original_grid_and_region_weighting(self):
        for dtype in (np.float16, np.float32, np.float64):
            masks, maps, _, _ = sample_data(dtype)
            for thresholds in (7, 200):
                with self.subTest(dtype=dtype, thresholds=thresholds):
                    self.assertAlmostEqual(compute_pro(masks, maps, thresholds),
                                           original_pro_with_compatible_apis(masks, maps, thresholds), places=13)

    def test_fragmented_masks_preserve_connectivity_and_uncached_aupro(self):
        masks = np.zeros((2, 15, 17), dtype=np.bool_)
        masks[0, ::2, ::2] = 1
        masks[1, 1::2, 1::2] = 1
        # One blank row/column separates every region even at connectivity=8.
        self.assertEqual(measure.label(masks[0]).max(), int(masks[0].sum()))
        maps = np.random.default_rng(14).random(masks.shape).astype(np.float32)
        expected = original_pro_with_compatible_apis(masks, maps, num_th=31)
        with mock.patch("rad_metrics.measure.regionprops", wraps=measure.regionprops) as regions:
            actual = compute_pro(masks, maps, num_th=31)
        self.assertAlmostEqual(actual, expected, places=13)
        self.assertEqual(regions.call_count, len(masks))
        self.assertTrue(all(call.kwargs.get("cache") is False for call in regions.call_args_list))

    def test_comparison_casting_and_near_equal_float32_values(self):
        for dtype in (np.float16, np.float32, np.float64):
            first = dtype(0.1)
            second = np.nextafter(first, dtype(np.inf))
            scores = np.array([first, first, second, dtype(0.9)], dtype=dtype)
            thresholds = np.array([float(first), (float(first) + float(second)) / 2,
                                   float(second), 0.9000000000001], dtype=np.float64)
            keys = _comparison_thresholds(scores.dtype, thresholds)
            observed = len(scores) - np.searchsorted(np.sort(scores), keys, side="right")
            expected = np.array([np.count_nonzero(scores > threshold) for threshold in thresholds])
            np.testing.assert_array_equal(observed, expected)

    def test_external_rankings_exact_ties_multiple_passes(self):
        rng = np.random.default_rng(8)
        for dtype in (np.float16, np.float32, np.float64):
            for chunk_size in (2, 17, 700):
                with self.subTest(dtype=dtype, chunk_size=chunk_size), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    labels = rng.integers(0, 2, size=283, dtype=np.uint8)
                    scores = (rng.integers(-4, 5, size=283) / 7).astype(dtype)
                    scores[:4] = np.array([-0.0, 0.0, 0.0, -0.0], dtype=dtype)
                    labels.tofile(root / "labels")
                    scores.tofile(root / "scores")
                    actual = _external_ranking_metrics(root / "labels", root / "scores", len(labels),
                                                       scores.dtype, root / "sort", chunk_size, lambda stage: None)
                    np.testing.assert_allclose(actual, sklearn_metrics(labels, scores), rtol=0, atol=2e-14)

    def test_rankings_do_not_downcast_float64(self):
        labels = np.array([0, 1, 0, 1], dtype=np.uint8)
        scores = np.array([1.0, 1.0 + 1e-12, 1.0 + 2e-12, 1.0 + 3e-12], dtype=np.float64)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels.tofile(root / "labels")
            scores.tofile(root / "scores")
            actual = _external_ranking_metrics(root / "labels", root / "scores", 4, scores.dtype,
                                               root / "sort", 1, lambda stage: None)
            np.testing.assert_allclose(actual, sklearn_metrics(labels, scores), rtol=0, atol=1e-14)

    def test_all_tied_rankings_match_sklearn(self):
        labels = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        scores = np.ones(5, dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels.tofile(root / "labels")
            scores.tofile(root / "scores")
            actual = _external_ranking_metrics(root / "labels", root / "scores", 5, scores.dtype,
                                               root / "sort", 1, lambda stage: None)
            np.testing.assert_allclose(actual, sklearn_metrics(labels, scores), rtol=0, atol=1e-14)

    def test_seven_metrics_batched_without_memmap(self):
        masks, maps, labels, scores = sample_data()
        expected = [*sklearn_metrics(labels, scores), *sklearn_metrics(masks.ravel(), maps.ravel()),
                    original_pro_with_compatible_apis(masks, maps)]
        with tempfile.TemporaryDirectory() as folder, mock.patch("numpy.memmap", side_effect=AssertionError("must not map the dataset")):
            with MetricAccumulator(Path(folder) / "spool", 6, 7, 9, chunk_size=3) as accumulator:
                for start, end in ((0, 1), (1, 4), (4, 6)):
                    accumulator.append(masks[start:end], maps[start:end], labels[start:end], scores[start:end])
                actual = accumulator.compute()
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-13)
                self.assertEqual(accumulator.compute(), actual)
            self.assertTrue((Path(folder) / "spool" / "metric-state.json").exists())

    def test_constant_or_single_class_inputs_explicitly_fail(self):
        masks, maps, labels, scores = sample_data()
        with self.assertRaisesRegex(ValueError, "non-constant"):
            compute_pro(masks, np.ones_like(maps))
        with self.assertRaisesRegex(ValueError, "both anomalous"):
            compute_pro(np.zeros_like(masks), maps)
        with self.assertRaisesRegex(ValueError, "both anomalous"):
            compute_pro(np.ones_like(masks), maps)
        with tempfile.TemporaryDirectory() as folder:
            with MetricAccumulator(folder, 6, 7, 9, chunk_size=7) as accumulator:
                accumulator.append(masks, maps, np.ones_like(labels), scores)
                with self.assertRaisesRegex(ValueError, "both positive and negative"):
                    accumulator.compute()
                self.assertIsNone(json.loads((Path(folder) / "metric-state.json").read_text())["results"])


class RecoveryTests(unittest.TestCase):
    def test_disk_preflight_retains_committed_samples_and_records_budget(self):
        masks, maps, labels, scores = sample_data()
        with tempfile.TemporaryDirectory() as folder:
            with MetricAccumulator(folder, 6, 7, 9, chunk_size=9) as accumulator:
                accumulator.append(masks[:1], maps[:1], labels[:1], scores[:1])
                with mock.patch("rad_metrics.shutil.disk_usage", return_value=mock.Mock(free=1)):
                    with self.assertRaisesRegex(OSError, "Insufficient metric-spool disk"):
                        accumulator.append(masks[1:], maps[1:], labels[1:], scores[1:])
                state = json.loads((Path(folder) / "metric-state.json").read_text())
                self.assertEqual(state["appended"], 1)
                self.assertEqual(state["disk_estimate"]["raw_total_bytes"], 6 * (7 * 9 + 1) * 5)
                self.assertGreater(state["disk_estimate"]["sort_additional_bytes"], 0)
            with MetricAccumulator.resume(folder) as accumulator:
                accumulator.append(masks[1:], maps[1:], labels[1:], scores[1:])
                self.assertEqual(len(accumulator.compute()), 7)

    def test_reserved_run_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            reserved = Path(folder) / "sort-pixel"
            reserved.mkdir()
            original = reserved / "caller-data.txt"
            original.write_text("preserve")
            with self.assertRaises(FileExistsError):
                MetricAccumulator(folder, 6, 7, 9)
            self.assertEqual(original.read_text(), "preserve")

    def test_resume_partial_collection_and_preserve_dtype(self):
        masks, maps, labels, scores = sample_data(np.float64)
        with tempfile.TemporaryDirectory() as folder:
            with MetricAccumulator(folder, 6, 7, 9, chunk_size=11) as accumulator:
                accumulator.append(masks[:2], maps[:2], labels[:2], scores[:2])
            # Simulate bytes written by a process killed before its batch commit.
            for filename in ("masks.bin", "maps.bin", "labels.bin", "scores.bin"):
                with open(Path(folder) / filename, "ab") as file:
                    file.write(b"uncommitted")
            with MetricAccumulator.resume(folder) as accumulator:
                self.assertEqual(accumulator.appended, 2)
                self.assertEqual((Path(folder) / "maps.bin").stat().st_size, 2 * 7 * 9 * 8)
                accumulator.append(masks[2:], maps[2:], labels[2:], scores[2:])
                result = accumulator.compute()
                self.assertEqual(len(result), 7)
                self.assertEqual(json.loads((Path(folder) / "metric-state.json").read_text())["map_dtype"], maps.dtype.str)

    def test_resource_stop_rolls_back_partial_batch(self):
        class StopGuard:
            count = 0
            def check(self):
                self.count += 1
                if self.count == 4:
                    raise RuntimeError("synthetic resource pressure")

        masks, maps, labels, scores = sample_data()
        with tempfile.TemporaryDirectory() as folder:
            with MetricAccumulator(folder, 6, 7, 9, chunk_size=9) as accumulator:
                accumulator.append(masks[:1], maps[:1], labels[:1], scores[:1])
                accumulator.guard = StopGuard()
                with self.assertRaisesRegex(RuntimeError, "synthetic resource pressure"):
                    accumulator.append(masks[1:3], maps[1:3], labels[1:3], scores[1:3])
                self.assertEqual(accumulator.appended, 1)
            state = json.loads((Path(folder) / "metric-state.json").read_text())
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["appended"], 1)
            self.assertEqual((Path(folder) / "maps.bin").stat().st_size, 7 * 9 * maps.dtype.itemsize)
            with MetricAccumulator.resume(folder) as accumulator:
                accumulator.append(masks[1:], maps[1:], labels[1:], scores[1:])
                self.assertEqual(len(accumulator.compute()), 7)

    def test_resource_stop_in_compute_can_resume_without_encoder(self):
        class StopGuard:
            def check(self):
                raise RuntimeError("synthetic resource pressure")

        masks, maps, labels, scores = sample_data()
        with tempfile.TemporaryDirectory() as folder:
            with MetricAccumulator(folder, 6, 7, 9, chunk_size=5) as accumulator:
                accumulator.append(masks, maps, labels, scores)
                accumulator.guard = StopGuard()
                with self.assertRaisesRegex(RuntimeError, "synthetic resource pressure"):
                    accumulator.compute()
            with MetricAccumulator.resume(folder) as accumulator:
                self.assertEqual(accumulator.appended, 6)
                self.assertEqual(len(accumulator.compute()), 7)
                accumulator.cleanup()
            self.assertEqual({path.name for path in Path(folder).iterdir()}, {".metric.lock"})

    def test_rejects_invalid_batches_and_incomplete_cleanup(self):
        masks, maps, labels, scores = sample_data()
        with tempfile.TemporaryDirectory() as folder:
            with MetricAccumulator(folder, 6, 7, 9) as accumulator:
                for invalid in (np.nan, np.inf):
                    bad = maps[:1].copy()
                    bad[0, 0, 0] = invalid
                    with self.assertRaisesRegex(ValueError, "finite"):
                        accumulator.append(masks[:1], bad, labels[:1], scores[:1])
                with self.assertRaisesRegex(ValueError, "binary"):
                    accumulator.append(np.full_like(masks[:1], 2, dtype=np.uint8), maps[:1], labels[:1], scores[:1])
                with self.assertRaisesRegex(ValueError, "successful compute"):
                    accumulator.cleanup()
                with self.assertRaisesRegex(ValueError, "Expected 6"):
                    accumulator.compute()
                self.assertEqual(accumulator.appended, 0)


class SpoolLockTests(unittest.TestCase):
    @staticmethod
    def independent_lock_attempt(directory):
        script = (
            "import fcntl, sys\n"
            "with open(sys.argv[1], 'a+b') as handle:\n"
            "    try:\n"
            "        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "    except BlockingIOError:\n"
            "        sys.exit(23)\n"
        )
        return subprocess.run([sys.executable, "-c", script, str(Path(directory) / ".metric.lock")],
                              check=False, capture_output=True, timeout=10).returncode

    def test_live_spool_rejects_create_and_resume_before_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            with MetricAccumulator(folder, 1, 1, 1):
                path = Path(folder) / "maps.bin"
                path.write_bytes(b"uncommitted tail")
                with self.assertRaisesRegex(RuntimeError, "Another process owns"):
                    MetricAccumulator(folder, 1, 1, 1)
                with self.assertRaisesRegex(RuntimeError, "Another process owns"):
                    MetricAccumulator.resume(folder)
                self.assertEqual(path.read_bytes(), b"uncommitted tail")
                self.assertEqual(self.independent_lock_attempt(folder), 23)
            self.assertEqual(self.independent_lock_attempt(folder), 0)
            with MetricAccumulator.resume(folder):
                self.assertEqual(path.stat().st_size, 0)

    def test_initialization_and_resume_failures_release_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(MetricAccumulator, "_save_state", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    MetricAccumulator(folder, 1, 1, 1)
            self.assertEqual(self.independent_lock_attempt(folder), 0)
            with MetricAccumulator(folder, 1, 1, 1):
                pass
            (Path(folder) / "metric-state.json").write_text('{"version": 99}')
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                MetricAccumulator.resume(folder)
            self.assertEqual(self.independent_lock_attempt(folder), 0)

    def test_context_exception_and_file_close_failure_release_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with MetricAccumulator(folder, 1, 1, 1):
                    raise RuntimeError("stop")
            self.assertEqual(self.independent_lock_attempt(folder), 0)
            accumulator = MetricAccumulator.resume(folder)
            broken = mock.Mock()
            broken.close.side_effect = OSError("close failure")
            accumulator._files["synthetic"] = broken
            with self.assertRaisesRegex(OSError, "close failure"):
                accumulator.close()
            self.assertTrue(accumulator._closed)
            self.assertEqual(accumulator._files, {})
            self.assertEqual(self.independent_lock_attempt(folder), 0)

    def test_compute_and_cleanup_hold_the_same_persistent_lock(self):
        masks, maps, labels, scores = sample_data()
        with tempfile.TemporaryDirectory() as folder:
            accumulator = MetricAccumulator(folder, 6, 7, 9)
            accumulator.append(masks, maps, labels, scores)
            inode = (Path(folder) / ".metric.lock").stat().st_ino

            def ranking(*args, **kwargs):
                self.assertEqual(self.independent_lock_attempt(folder), 23)
                return (0.5, 0.5, 0.5)

            with mock.patch("rad_metrics._aupro_from_images", return_value=0.5), \
                    mock.patch("rad_metrics._external_ranking_metrics", side_effect=ranking):
                self.assertEqual(accumulator.compute(), [0.5] * 7)
            original_unlink = Path.unlink

            def unlink(path, *args, **kwargs):
                if path.name == "maps.bin":
                    self.assertEqual(self.independent_lock_attempt(folder), 23)
                return original_unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", unlink):
                accumulator.cleanup()
            self.assertEqual({path.name for path in Path(folder).iterdir()}, {".metric.lock"})
            self.assertEqual((Path(folder) / ".metric.lock").stat().st_ino, inode)
            self.assertEqual(self.independent_lock_attempt(folder), 0)

    def test_cleanup_failure_releases_lock_and_keeps_state(self):
        with tempfile.TemporaryDirectory() as folder:
            accumulator = MetricAccumulator(folder, 1, 1, 1)
            accumulator._state["results"] = [0.5] * 7
            accumulator._save_state()
            with mock.patch.object(Path, "unlink", side_effect=OSError("cleanup failure")):
                with self.assertRaisesRegex(OSError, "cleanup failure"):
                    accumulator.cleanup()
            self.assertTrue((Path(folder) / "metric-state.json").exists())
            self.assertEqual(self.independent_lock_attempt(folder), 0)
            with self.assertRaisesRegex(ValueError, "resume before cleanup"):
                accumulator.cleanup()


if __name__ == "__main__":
    unittest.main()
