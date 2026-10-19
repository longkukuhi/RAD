"""Disk-backed, bounded-memory implementations of RAD's seven metrics.

Ranking is exact (including score ties); input floating-point dtypes are retained.
The historical AUPRO threshold grid and FPR normalization are intentionally kept.
No model, torch, pandas, or whole-dataset sklearn operation is used here.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
from typing import Callable

import numpy as np
from skimage import measure


DEFAULT_CHUNK_SIZE = 262_144  # score/label records, not images
MERGE_FAN_IN = 32
MERGE_BUFFER_RECORDS = 4096
DISK_RESERVE_BYTES = 64 * 1024**2
_FLOAT_DTYPES = {np.dtype("float16"), np.dtype("float32"), np.dtype("float64")}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _check(guard):
    if guard is not None:
        guard.check()


def _binary(array, name):
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary 0/1 values")


def _score_dtype(array, name):
    if array.dtype not in _FLOAT_DTYPES:
        raise ValueError(f"{name} must use float16, float32, or float64; got {array.dtype}")
    return array.dtype


def _record_dtype(score_dtype):
    return np.dtype([("score", score_dtype), ("positive", "<i8"), ("negative", "<i8")])


def _remove_derived_tree(directory):
    """Remove generated runs without listing every run name in memory."""
    directory = Path(directory)
    if directory.is_symlink():
        directory.unlink()
        return
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                _remove_derived_tree(entry.path)
            else:
                os.unlink(entry.path)
    directory.rmdir()


def _aggregate_sorted(records):
    """Sort a bounded record window, returning one record per distinct score."""
    if not len(records):
        return records
    order = np.argsort(records["score"], kind="stable")[::-1]
    records = records[order]
    starts = np.r_[0, np.flatnonzero(records["score"][1:] != records["score"][:-1]) + 1]
    grouped = np.empty(len(starts), dtype=records.dtype)
    grouped["score"] = records["score"][starts]
    grouped["positive"] = np.add.reduceat(records["positive"], starts)
    grouped["negative"] = np.add.reduceat(records["negative"], starts)
    return grouped


class _RunReader:
    def __init__(self, path, dtype, buffer_records):
        self.file = open(path, "rb")
        self.dtype = dtype
        self.buffer_records = buffer_records
        self.buffer = np.empty(0, dtype=dtype)
        self.position = 0

    def available(self):
        if self.position == len(self.buffer):
            self.buffer = np.fromfile(self.file, dtype=self.dtype, count=self.buffer_records)
            self.position = 0
        return self.buffer[self.position:]

    def close(self):
        self.file.close()


def _merge_runs(paths, destination, dtype, buffer_records, checkpoint):
    """Vectorized bounded multiway merge, without a Python loop per pixel.

    Each input already has strictly decreasing, unique scores. A window's last
    score bounds its unread tail. Everything >= the greatest such bound can be
    emitted together, including complete ties across all inputs.
    """
    readers = []
    try:
        for path in paths:
            readers.append(_RunReader(path, dtype, buffer_records))
        with open(destination, "wb") as output:
            while True:
                checkpoint("merge ranking runs")
                windows = [(reader, reader.available()) for reader in readers]
                windows = [(reader, window) for reader, window in windows if len(window)]
                if not windows:
                    break
                boundary = max(window["score"][-1] for _, window in windows)
                prefixes = []
                for reader, window in windows:
                    # Negation preserves finite floating-point ordering and ties.
                    count = int(np.searchsorted(-window["score"], -boundary, side="right"))
                    if count:
                        prefixes.append(window[:count])
                        reader.position += count
                combined = np.concatenate(prefixes)
                _aggregate_sorted(combined).tofile(output)
    finally:
        for reader in readers:
            reader.close()


def _external_ranking_metrics(label_path, score_path, count, score_dtype,
                             directory, chunk_size, checkpoint):
    """AUROC, average precision and historical maximum F1 from disk records."""
    directory = Path(directory)
    # Derived runs may be rebuilt after interruption; the original spool is kept.
    if directory.exists():
        _remove_derived_tree(directory)
    directory.mkdir()
    run_dir = directory / "pass-0"
    run_dir.mkdir()
    dtype = _record_dtype(score_dtype)
    total_positive = 0
    run_count = 0
    with open(label_path, "rb") as labels_file, open(score_path, "rb") as scores_file:
        remaining = count
        while remaining:
            checkpoint("sort ranking chunk")
            size = min(chunk_size, remaining)
            labels = np.fromfile(labels_file, dtype=np.uint8, count=size)
            scores = np.fromfile(scores_file, dtype=score_dtype, count=size)
            if len(labels) != size or len(scores) != size:
                raise ValueError("Metric spool is truncated")
            records = np.empty(size, dtype=dtype)
            records["score"] = scores
            records["positive"] = labels
            records["negative"] = 1 - labels
            total_positive += int(labels.sum(dtype=np.int64))
            _aggregate_sorted(records).tofile(run_dir / f"{run_count}.bin")
            run_count += 1
            remaining -= size
    total_negative = count - total_positive
    if not total_positive or not total_negative:
        raise ValueError("AUROC requires both positive and negative labels; seven metrics are not available")

    pass_number = 0
    buffer_records = min(chunk_size, MERGE_BUFFER_RECORDS)
    while run_count > 1:
        pass_number += 1
        next_dir = directory / f"pass-{pass_number}"
        next_dir.mkdir()
        next_count = 0
        for start in range(0, run_count, MERGE_FAN_IN):
            paths = [run_dir / f"{i}.bin" for i in range(start, min(start + MERGE_FAN_IN, run_count))]
            _merge_runs(paths, next_dir / f"{next_count}.bin", dtype, buffer_records, checkpoint)
            for path in paths:
                path.unlink()
            next_count += 1
        run_dir.rmdir()
        run_dir, run_count = next_dir, next_count

    true_positive = false_positive = 0
    previous_recall = previous_fpr = 0.0
    # Compensated scalar sums do not retain a list proportional to data size.
    roc_sum = roc_correction = ap_sum = ap_correction = 0.0
    best_f1 = 0.0
    with open(run_dir / "0.bin", "rb") as final_run:
        while True:
            checkpoint("accumulate ranking metrics")
            records = np.fromfile(final_run, dtype=dtype, count=chunk_size)
            if not len(records):
                break
            tp = np.cumsum(records["positive"], dtype=np.int64) + true_positive
            fp = np.cumsum(records["negative"], dtype=np.int64) + false_positive
            recall = tp / total_positive
            fpr = fp / total_negative
            precision = tp / (tp + fp)
            prior_recall = np.r_[previous_recall, recall[:-1]]
            prior_fpr = np.r_[previous_fpr, fpr[:-1]]
            roc_piece = float(np.sum((fpr - prior_fpr) * (recall + prior_recall) * 0.5))
            ap_piece = float(np.sum((recall - prior_recall) * precision))
            roc_sum, roc_correction = _compensated_add(roc_sum, roc_correction, roc_piece)
            ap_sum, ap_correction = _compensated_add(ap_sum, ap_correction, ap_piece)
            # Preserve utils.f1_score_max's epsilon, not a different F1 formula.
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            best_f1 = max(best_f1, float(f1.max()))
            true_positive, false_positive = int(tp[-1]), int(fp[-1])
            previous_recall, previous_fpr = float(recall[-1]), float(fpr[-1])
    return float(roc_sum + roc_correction), float(ap_sum + ap_correction), best_f1


def _compensated_add(total, correction, value):
    updated = total + value
    if np.ndim(updated):
        correction = correction + np.where(np.abs(total) >= np.abs(value),
                                            (total - updated) + value,
                                            (value - updated) + total)
    elif abs(total) >= abs(value):
        correction += (total - updated) + value
    else:
        correction += (value - updated) + total
    return updated, correction


def _comparison_thresholds(score_dtype, thresholds):
    """Match array > numpy-scalar promotion in both NumPy 1.x and NumPy 2.x.

    searchsorted with a vector of float64 thresholds otherwise differs from the
    old loop on NumPy versions with value-dependent scalar casting.
    """
    probe = np.empty(0, dtype=score_dtype)
    return np.asarray([np.asarray(th, dtype=np.result_type(probe, th)).item()
                       for th in thresholds], dtype=np.float64)


def _aupro_from_images(image_factory: Callable, minimum, maximum, num_th=200,
                       checkpoint=lambda stage: None):
    num_th = _positive_int(num_th, "num_th")
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError("AUPRO requires finite, non-constant anomaly maps")
    # Do not replace this grid with linspace, quantiles, or a different endpoint.
    delta = (maximum - minimum) / num_th
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("AUPRO threshold spacing is not finite and positive in the input dtype")
    thresholds = np.arange(minimum, maximum, delta)
    if len(thresholds) < 2:
        raise ValueError("AUPRO requires at least two thresholds")
    fp = np.zeros(len(thresholds), dtype=np.int64)
    pro_sum = np.zeros(len(thresholds), dtype=np.float64)
    pro_correction = np.zeros(len(thresholds), dtype=np.float64)
    negative_pixels = 0
    region_count = 0
    for mask, anomaly_map in image_factory():
        checkpoint("accumulate AUPRO image")
        _binary(mask, "ground-truth masks")
        if not np.isfinite(anomaly_map).all():
            raise ValueError("Anomaly maps must be finite")
        mask = mask.astype(np.bool_, copy=False)
        components = measure.label(mask)  # same default connectivity as the old code
        keys = _comparison_thresholds(anomaly_map.dtype, thresholds)
        background = np.sort(anomaly_map[~mask], axis=None)
        negative_pixels += len(background)
        fp += len(background) - np.searchsorted(background, keys, side="right")
        del background
        # RegionProperties objects share this one label image. Do not retain
        # each accessed region's image/area cache for the rest of the image.
        for region in measure.regionprops(components, cache=False):
            if region_count % 128 == 0:
                checkpoint("accumulate AUPRO regions")
            coordinates = region.coords
            region_scores = np.sort(anomaly_map[coordinates[:, 0], coordinates[:, 1]], axis=None)
            hits = len(region_scores) - np.searchsorted(region_scores, keys, side="right")
            # region.area and the region weighting match the original PRO mean.
            contribution = hits / region.area
            pro_sum, pro_correction = _compensated_add(pro_sum, pro_correction, contribution)
            region_count += 1
    if not region_count or not negative_pixels:
        raise ValueError("AUPRO requires both anomalous regions and background pixels")
    fpr = fp / negative_pixels
    pro = (pro_sum + pro_correction) / region_count
    keep = fpr < 0.3
    fpr, pro = fpr[keep], pro[keep]
    if len(fpr) < 2 or fpr.max() <= 0:
        raise ValueError("AUPRO has fewer than two usable thresholds or zero retained FPR range")
    # Preserve historical normalization by the largest retained FPR, not by 0.3.
    fpr = fpr / fpr.max()
    dx = np.diff(fpr)
    if np.any(dx > 0):
        raise ValueError("AUPRO FPR sequence is not non-increasing")
    return float(-np.sum(dx * (pro[1:] + pro[:-1]) * 0.5))


def compute_pro(masks, amaps, num_th=200):
    """Compatible array entry point; working arrays are bounded to one image.

    For datasets that do not already fit RAM, use MetricAccumulator instead.
    Undefined inputs raise ValueError instead of returning fabricated metrics.
    """
    if not isinstance(masks, np.ndarray) or not isinstance(amaps, np.ndarray):
        raise ValueError("masks and amaps must be numpy arrays")
    if masks.ndim != 3 or amaps.shape != masks.shape or not len(masks):
        raise ValueError("masks and amaps must share non-empty [N,H,W] shapes")
    _score_dtype(amaps, "anomaly maps")
    minimum = maximum = None
    for mask, amap in zip(masks, amaps):
        _binary(mask, "ground-truth masks")
        if not np.isfinite(amap).all():
            raise ValueError("Anomaly maps must be finite")
        lo, hi = amap.min(), amap.max()
        minimum = lo if minimum is None or lo < minimum else minimum
        maximum = hi if maximum is None or hi > maximum else maximum
    return _aupro_from_images(lambda: zip(masks, amaps), minimum, maximum, num_th)


class MetricAccumulator:
    """Append CPU batches to a recoverable disk spool and compute RAD metrics.

    ``chunk_size`` caps score records in an initial sort chunk, independently of
    dataset size. AUPRO additionally needs one H x W image and its components.
    ``directory`` is an exclusive spool directory; existing spools require
    :meth:`resume`. ``close`` and context-manager exit NEVER remove evidence.
    After successful computation callers may explicitly invoke ``cleanup``.
    A process lock protects the entire open lifetime, including computation and
    cleanup. The lock file and its directory remain after cleanup so another
    opener can never acquire a different lock inode for the same spool.
    """

    _FILES = {"masks": "masks.bin", "maps": "maps.bin", "labels": "labels.bin", "scores": "scores.bin"}
    _SORT_DIRS = ("sort-image", "sort-pixel")
    _LOCK_FILE = ".metric.lock"

    def __init__(self, directory, num_samples, height, width,
                 chunk_size=DEFAULT_CHUNK_SIZE, guard=None):
        self.directory = Path(directory)
        self.num_samples = _positive_int(num_samples, "num_samples")
        self.height = _positive_int(height, "height")
        self.width = _positive_int(width, "width")
        self.chunk_size = _positive_int(chunk_size, "chunk_size")
        if self.num_samples * self.height * self.width > np.iinfo(np.int64).max:
            raise ValueError("Pixel count exceeds exact int64 counters")
        self.guard = guard
        self._files = {}
        self._closed = False
        self._lock_handle = None
        self.directory.mkdir(parents=True, exist_ok=True)
        self._state_path = self.directory / "metric-state.json"
        self._acquire_lock()
        try:
            reserved = (*self._FILES.values(), *self._SORT_DIRS, "metric-state.json", "metric-state.json.tmp")
            if any((self.directory / name).exists() or (self.directory / name).is_symlink() for name in reserved):
                raise FileExistsError(f"Metric spool already exists; use MetricAccumulator.resume({str(self.directory)!r})")
            self._state = {"version": 1, "num_samples": self.num_samples, "height": self.height,
                           "width": self.width, "chunk_size": self.chunk_size, "appended": 0,
                           "map_dtype": None, "score_dtype": None, "minimum": None,
                           "maximum": None, "status": "collecting", "results": None}
            self._save_state()
            self._open_files("w+b")
        except BaseException:
            self.close()
            raise

    @classmethod
    def resume(cls, directory, *, chunk_size=None, guard=None):
        """Reopen committed batches; any uncommitted partial batch is discarded.

        Derived sorting runs are rebuilt by compute; source maps/labels are kept.
        Resume does not rerun the encoder. The caller must skip ``appended``
        already committed examples if additional batches are still needed.
        """
        self = cls.__new__(cls)
        self.directory = Path(directory)
        self._state_path = self.directory / "metric-state.json"
        self.guard = guard
        self._closed = False
        self._files = {}
        self._lock_handle = None
        self._acquire_lock()
        try:
            with open(self._state_path, encoding="utf-8") as stream:
                self._state = json.load(stream)
            if self._state.get("version") != 1:
                raise ValueError("Unsupported metric spool version")
            for name in ("num_samples", "height", "width"):
                setattr(self, name, _positive_int(self._state[name], name))
            self.chunk_size = _positive_int(self._state["chunk_size"] if chunk_size is None else chunk_size, "chunk_size")
            self._open_files("r+b")
            counts = {"masks": self.appended * self.height * self.width,
                      "maps": self.appended * self.height * self.width * self._dtype_size("map_dtype"),
                      "labels": self.appended,
                      "scores": self.appended * self._dtype_size("score_dtype")}
            for name, size in counts.items():
                file = self._files[name]
                if os.fstat(file.fileno()).st_size < size:
                    raise ValueError(f"Committed metric spool file {name} is truncated")
                file.truncate(size)
                file.seek(size)
        except BaseException:
            self.close()
            raise
        return self

    def _acquire_lock(self):
        import fcntl
        handle = open(self.directory / self._LOCK_FILE, "a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as exc:
            handle.close()
            if isinstance(exc, BlockingIOError):
                raise RuntimeError("Another process owns this metric spool; wait for it to close before resuming") from exc
            raise
        self._lock_handle = handle

    @property
    def appended(self):
        return int(self._state["appended"])

    def _dtype_size(self, field):
        return 0 if self._state[field] is None else np.dtype(self._state[field]).itemsize

    def _open_files(self, mode):
        try:
            for name, filename in self._FILES.items():
                self._files[name] = open(self.directory / filename, mode)
        except BaseException:
            self.close()
            raise

    def _save_state(self):
        temporary = self._state_path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(self._state, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self._state_path)

    def _checkpoint(self, stage):
        try:
            _check(self.guard)
        except Exception as exc:
            self._record_failure(stage, exc)
            raise

    def _record_failure(self, stage, exc):
        self._state["status"] = "interrupted"
        self._state["failure"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        self._save_state()

    def _ensure_disk_space(self, map_dtype, score_dtype, *, collecting):
        """Reserve space for all declared samples and two sorting generations.

        The estimate deliberately assumes every score is distinct. Disk files
        scale with N; RAM does not. Completed source batches are never deleted
        when capacity is insufficient. File-allocation slack is included for
        small sort chunks as well as the 64 MiB safety reserve.
        """
        pixels = self.num_samples * self.height * self.width
        raw_total = pixels * (1 + map_dtype.itemsize) + self.num_samples * (1 + score_dtype.itemsize)
        raw_remaining = ((self.num_samples - self.appended)
                         * (self.height * self.width * (1 + map_dtype.itemsize) + 1 + score_dtype.itemsize))
        filesystem = os.statvfs(self.directory)
        allocation_unit = max(filesystem.f_bsize, filesystem.f_frsize, 4096)
        sort_bytes = 0
        for count, dtype in ((pixels, map_dtype), (self.num_samples, score_dtype)):
            run_count = (count + self.chunk_size - 1) // self.chunk_size
            # Two generations coexist during merging. The final image ranking
            # also remains while pixel ranking is computed, covered by this sum.
            sort_bytes += 2 * (count * _record_dtype(dtype).itemsize + run_count * allocation_unit)
        required = (raw_remaining if collecting else 0) + sort_bytes + DISK_RESERVE_BYTES
        available = shutil.disk_usage(self.directory).free
        self._state["disk_estimate"] = {
            "raw_total_bytes": raw_total, "raw_remaining_bytes": raw_remaining if collecting else 0,
            "sort_additional_bytes": sort_bytes, "reserve_bytes": DISK_RESERVE_BYTES,
            "required_free_bytes": required, "available_bytes": available,
            "stage": "collect" if collecting else "compute", "chunk_size": self.chunk_size,
        }
        self._save_state()
        if available < required:
            exc = OSError(
                f"Insufficient metric-spool disk space: need {required} free bytes "
                f"for remaining raw data, exact sorting and reserve; have {available}. "
                "Committed samples remain recoverable.")
            self._record_failure("metric disk capacity", exc)
            raise exc

    def append(self, gt_cpu_numpy, maps, labels, scores):
        if self._closed:
            raise ValueError("MetricAccumulator is closed")
        arrays = (gt_cpu_numpy, maps, labels, scores)
        if not all(isinstance(array, np.ndarray) for array in arrays):
            raise ValueError("append expects four CPU numpy arrays")
        expected_tail = (self.height, self.width)
        if maps.ndim != 3 or maps.shape[1:] != expected_tail or gt_cpu_numpy.shape != maps.shape:
            raise ValueError(f"Masks and maps must have shape [B,{self.height},{self.width}]")
        size = len(maps)
        if not size or labels.shape != (size,) or scores.shape != (size,):
            raise ValueError("labels and scores must have shape [B] for a non-empty batch")
        if self.appended + size > self.num_samples or self._state["results"] is not None:
            raise ValueError("Append exceeds the declared sample count or a completed spool")
        map_dtype, score_dtype = _score_dtype(maps, "maps"), _score_dtype(scores, "scores")
        if self._state["map_dtype"] is not None and (
            map_dtype != np.dtype(self._state["map_dtype"]) or score_dtype != np.dtype(self._state["score_dtype"])
        ):
            raise ValueError("Floating-point dtypes must remain unchanged across appended batches")
        # Validate before writing; only one image-sized temporary at a time.
        _binary(labels, "image labels")
        if not np.isfinite(scores).all():
            raise ValueError("Image scores must be finite")
        minimum = None if self._state["minimum"] is None else map_dtype.type(self._state["minimum"])
        maximum = None if self._state["maximum"] is None else map_dtype.type(self._state["maximum"])
        for mask, amap in zip(gt_cpu_numpy, maps):
            self._checkpoint("validate metric batch")
            _binary(mask, "ground-truth masks")
            if not np.isfinite(amap).all():
                raise ValueError("Anomaly maps must be finite")
            lo, hi = amap.min(), amap.max()
            minimum = lo if minimum is None or lo < minimum else minimum
            maximum = hi if maximum is None or hi > maximum else maximum
        self._ensure_disk_space(map_dtype, score_dtype, collecting=True)
        offsets = {name: file.tell() for name, file in self._files.items()}
        committed_state = dict(self._state)
        try:
            for mask, amap in zip(gt_cpu_numpy, maps):
                self._checkpoint("write metric image")
                mask.astype(np.uint8, copy=False).tofile(self._files["masks"])
                amap.tofile(self._files["maps"])
            labels.astype(np.uint8, copy=False).tofile(self._files["labels"])
            scores.tofile(self._files["scores"])
            for file in self._files.values():
                file.flush()
                os.fsync(file.fileno())
            self._state.update(appended=self.appended + size, map_dtype=map_dtype.str,
                               score_dtype=score_dtype.str, minimum=float(minimum), maximum=float(maximum),
                               status="collecting")
            self._state.pop("failure", None)
            self._save_state()
        except BaseException as exc:
            # Keep committed batches and a receipt. A later resume also truncates
            # crash tails using the atomically committed sample count.
            for name, file in self._files.items():
                file.seek(offsets[name])
                file.truncate()
            self._state = committed_state
            self._record_failure("append metric batch", exc)
            raise

    def _images(self):
        pixels = self.height * self.width
        dtype = np.dtype(self._state["map_dtype"])
        with open(self.directory / "masks.bin", "rb") as masks_file, open(self.directory / "maps.bin", "rb") as maps_file:
            for _ in range(self.appended):
                mask = np.fromfile(masks_file, dtype=np.uint8, count=pixels)
                amap = np.fromfile(maps_file, dtype=dtype, count=pixels)
                if len(mask) != pixels or len(amap) != pixels:
                    raise ValueError("Metric spool is truncated")
                yield mask.reshape(self.height, self.width), amap.reshape(self.height, self.width)

    def compute(self):
        if self._closed:
            raise ValueError("MetricAccumulator is closed; reopen with resume")
        if self.appended != self.num_samples:
            raise ValueError(f"Expected {self.num_samples} samples, have {self.appended}")
        if self._state["results"] is not None:
            return list(self._state["results"])
        self._state["status"] = "computing"
        self._save_state()
        try:
            self._checkpoint("start metrics")
            map_dtype = np.dtype(self._state["map_dtype"])
            # An explicit compute retry rebuilds derived runs, reclaiming their
            # space before the capacity check. Committed source files stay intact.
            for name in self._SORT_DIRS:
                if (self.directory / name).exists():
                    _remove_derived_tree(self.directory / name)
            self._ensure_disk_space(map_dtype, np.dtype(self._state["score_dtype"]), collecting=False)
            aupro = _aupro_from_images(self._images, map_dtype.type(self._state["minimum"]),
                                       map_dtype.type(self._state["maximum"]), checkpoint=self._checkpoint)
            image_metrics = _external_ranking_metrics(
                self.directory / "labels.bin", self.directory / "scores.bin", self.num_samples,
                np.dtype(self._state["score_dtype"]), self.directory / "sort-image", self.chunk_size, self._checkpoint)
            pixel_metrics = _external_ranking_metrics(
                self.directory / "masks.bin", self.directory / "maps.bin", self.num_samples * self.height * self.width,
                map_dtype, self.directory / "sort-pixel", self.chunk_size, self._checkpoint)
            results = [*image_metrics, *pixel_metrics, aupro]
            if not all(math.isfinite(value) for value in results):
                raise ValueError("Metric computation produced non-finite results")
            self._state.update(status="complete", results=results)
            self._state.pop("failure", None)
            self._save_state()
            return list(results)
        except BaseException as exc:
            # Resource checkpoints already recorded the precise failed stage.
            # Also retain a receipt for Ctrl-C or any non-checkpoint failure.
            failure = self._state.get("failure", {})
            stage = (failure.get("stage", "compute metrics")
                     if self._state["status"] == "interrupted" else "compute metrics")
            self._record_failure(stage, exc)
            raise

    def _close_files(self):
        first_error = None
        for file in self._files.values():
            try:
                file.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._files.clear()
        if first_error is not None:
            raise first_error

    def close(self):
        try:
            self._close_files()
        finally:
            try:
                if self._lock_handle is not None:
                    self._lock_handle.close()
                    self._lock_handle = None
            finally:
                self._closed = True

    def cleanup(self):
        """Remove completed spool data under its lock; retain the lock inode."""
        if self._closed:
            raise ValueError("MetricAccumulator is closed; resume before cleanup")
        if self._state.get("results") is None:
            raise ValueError("Keep incomplete spools for recovery; cleanup requires successful compute")
        try:
            self._close_files()
            for filename in self._FILES.values():
                (self.directory / filename).unlink(missing_ok=True)
            for name in self._SORT_DIRS:
                if (self.directory / name).exists():
                    _remove_derived_tree(self.directory / name)
            self._state_path.unlink(missing_ok=True)
            self._state_path.with_suffix(".json.tmp").unlink(missing_ok=True)
            # Never unlink the lock or remove its directory: a waiting opener may
            # already hold this inode. Caller-owned files also remain untouched.
        finally:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
