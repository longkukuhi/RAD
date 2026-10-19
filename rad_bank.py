"""Versioned, resumable float32 feature banks with bounded explicit disk reads.

No bank is memory-mapped and no legacy checkpoint is implicitly loaded.  A bank
is usable only after manifest.json is atomically published.  Each append commits
all layers and its reference records together in .progress.json; --resume-bank
can validate and continue an interrupted build without changing reference order.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zipfile

import numpy as np
import torch

VERSION = 2
IO_CHUNK = 1024 * 1024
DEFAULT_MAX_READ_BYTES = 128 * 1024 * 1024


class BankError(ValueError):
    """A bank is incomplete, incompatible, corrupt, or exceeds its I/O budget."""


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode('utf-8')


def _fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _hash_prefix(path, length=None, guard=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        remaining = length
        while remaining is None or remaining > 0:
            block = handle.read(IO_CHUNK if remaining is None else min(IO_CHUNK, remaining))
            if not block:
                if remaining:
                    raise BankError(f'Truncated bank file: {path}')
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
            del block
            if guard is not None:
                guard.check()
    return digest


def checkpoint_fingerprint(path):
    """SHA256 with a one-MiB buffer; never deserialize a model checkpoint."""
    return _hash_prefix(path).hexdigest()


def file_fingerprint(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = checkpoint_fingerprint(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise BankError(f'Reference changed while hashing: {path}')
    return {'path': str(path), 'size': before.st_size, 'mtime_ns': before.st_mtime_ns, 'sha256': digest}


def dataset_reference(dataset, index, categories, data_root):
    """Fingerprint one reference without loading/decompressing its image."""
    base = getattr(dataset, 'base', dataset)
    category_index = bisect.bisect_right(base.cumulative_sizes, index)
    offset = 0 if category_index == 0 else base.cumulative_sizes[category_index - 1]
    sample_index = index - offset
    child = base.datasets[category_index]
    if hasattr(child, 'samples'):
        path = child.samples[sample_index][0]
    else:
        path = child.img_paths[sample_index]
    record = file_fingerprint(path)
    # Header inspection does not decode the image. This lets builders budget
    # actual source-resolution decoding instead of assuming all inputs are 512².
    from PIL import Image
    with Image.open(path) as image:
        record['image_width'], record['image_height'] = image.size
    record.update(index=index, category=str(categories[category_index]), category_index=category_index,
                  sample_index=sample_index, relative_path=os.path.relpath(record['path'], Path(data_root).resolve()))
    return record


def build_provenance(*, encoder_name, checkpoint_path, image_size, crop_size, categories,
                     reference_records, source_file=None, guard=None):
    """Stream source fingerprints in exact dataset order; retain no record list."""
    digest = hashlib.sha256()
    count = 0
    max_source_pixels = 0
    for count, record in enumerate(reference_records, start=1):
        if record.get('index') != count - 1:
            raise BankError('Reference indices must be contiguous and start at zero')
        digest.update(_json_bytes(record))
        max_source_pixels = max(max_source_pixels, int(record.get('image_width', 0)) * int(record.get('image_height', 0)))
        if guard is not None:
            guard.check()
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unknown'
    source = {'commit': revision, 'rad_bank_sha256': checkpoint_fingerprint(__file__)}
    if source_file is not None:
        source['builder_sha256'] = checkpoint_fingerprint(source_file)
    return {
        'encoder': {'name': encoder_name, 'checkpoint_sha256': checkpoint_fingerprint(checkpoint_path)},
        'preprocessing': {'image_size': int(image_size), 'crop_size': int(crop_size),
                          'resize': 'square', 'interpolation': 'bilinear',
                          'order': ['Resize', 'ToTensor', 'CenterCrop', 'Normalize'],
                          'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'categories': list(categories), 'references_sha256': digest.hexdigest(),
        'reference_count': count, 'source_revision': source,
        'max_source_image_pixels': max_source_pixels,
        'feature_dtype': 'float32', 'seed': 1,
    }


def estimate_build_resources(*, encoder_name, checkpoint_path, batch_size, image_size, crop_size,
                             layers, num_workers, device, max_source_pixels=0):
    """Conservative analytic peak, not a measurement; CLI measured overrides win.

    Architecture constants mirror dinov3/hub/backbones.py. No backward graph is
    retained. Include the SDPA math fallback (two B*heads*T*T float32 matrices),
    QKV/residual/MLP workspaces, both intermediate and normalized layer outputs,
    and actual checkpoint storage. DataLoader prefetch_factor is fixed at one.
    """
    architectures = {
        'dinov3_vits16': (384, 12, 6, 4, 2),
        'dinov3_vits16plus': (384, 12, 6, 6, 3),
        'dinov3_vitb16': (768, 12, 12, 4, 2),
        'dinov3_vitl16': (1024, 24, 16, 4, 2),
        'dinov3_vitl16plus': (1024, 24, 16, 6, 3),
        'dinov3_vith16plus': (1280, 32, 20, 6, 3),
        'dinov3_vit7b16': (4096, 40, 32, 3, 3),
    }
    if encoder_name not in architectures:
        raise BankError(f'No verified extraction resource model for encoder {encoder_name}')
    if batch_size < 1 or num_workers < 0 or min(image_size, crop_size) < 1 or crop_size % 16:
        raise BankError('Invalid batch/worker/image dimensions for the patch16 resource model')
    channels, depth, heads, ratio, ffn_projections = architectures[encoder_name]
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0 or max(layers) >= depth:
        raise BankError('Extraction layer indices are outside the selected encoder')
    tokens = (crop_size // 16) ** 2 + 5  # CLS plus four storage tokens.
    token_bytes = batch_size * tokens * channels * 4
    image_batch = batch_size * crop_size ** 2 * 3 * 4
    checkpoint_bytes = Path(checkpoint_path).stat().st_size
    if zipfile.is_zipfile(checkpoint_path):
        with zipfile.ZipFile(checkpoint_path) as archive:
            checkpoint_bytes = sum(info.file_size for info in archive.infolist())
    # Biases, norms, RoPE, patch embedding and register tokens are also counted.
    parameter_bytes = int(4 * (depth * ((4 + ffn_projections * ratio) * channels ** 2 + 32 * channels)
                                 + 3 * 16 ** 2 * channels + 32 * channels))
    weight_bytes = max(parameter_bytes, checkpoint_bytes)
    retained_outputs = 2 * len(layers) * token_bytes
    attention_matrices = 2 * batch_size * heads * tokens ** 2 * 4
    forward_workspace = attention_matrices + (8 + 2 * ratio) * token_bytes + retained_outputs
    cpu_layer_copy = batch_size * (crop_size // 16) ** 2 * channels * 4
    # Each worker may hold a decoded source image, a resized tensor, and its
    # one queued batch. Two more batches cover collation/main-process transfer.
    source_pixels = max(max_source_pixels, image_size ** 2)
    decoder_bytes = max(num_workers, 1) * (12 * source_pixels + 12 * image_size ** 2)
    prefetch_bytes = (num_workers + 2) * image_batch
    runtime_bytes = (256 + 64 * num_workers) * 1024 ** 2
    host_base = 2 * weight_bytes + checkpoint_bytes + cpu_layer_copy + decoder_bytes + prefetch_bytes + runtime_bytes
    if not str(device).startswith('cuda'):
        host_base += forward_workspace
    ram = math.ceil(host_base * 1.25)  # allocator/Python/serialization margin.
    gpu = math.ceil((weight_bytes + image_batch + forward_workspace) * 1.25) + 256 * 1024 ** 2 if str(device).startswith('cuda') else 0
    return {'ram_budget_bytes': ram, 'gpu_budget_bytes': gpu,
            'basis': 'analytic float32 no-grad upper estimate; not measured',
            'components': {'batch_size': batch_size, 'tokens': tokens, 'channels': channels,
                           'depth': depth, 'heads': heads, 'weight_bytes': weight_bytes,
                           'checkpoint_bytes': checkpoint_bytes, 'attention_matrices_bytes': attention_matrices,
                           'forward_workspace_bytes': forward_workspace, 'prefetch_bytes': prefetch_bytes,
                           'decoder_bytes': decoder_bytes, 'cpu_layer_copy_bytes': cpu_layer_copy,
                           'runtime_bytes': runtime_bytes, 'allocator_margin': 1.25}}


def _validate_expected(actual, expected, prefix='provenance'):
    for key, value in expected.items():
        if key not in actual:
            raise BankError(f'Missing {prefix}.{key}')
        if isinstance(value, dict):
            if not isinstance(actual[key], dict):
                raise BankError(f'Mismatched {prefix}.{key}')
            _validate_expected(actual[key], value, prefix + '.' + key)
        elif actual[key] != value:
            raise BankError(f'Mismatched {prefix}.{key}: bank does not match requested inputs')


def _safe_file(root, filename):
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise BankError('Bank data files must be direct children of the bank directory')
    path = root / filename
    if path.is_symlink():
        raise BankError(f'Symlink is not allowed for bank data: {filename}')
    return path


def _write_all(handle, data):
    view = memoryview(data).cast('B')
    while view:
        written = handle.write(view)
        if not written:
            raise OSError('Short write while saving feature bank')
        view = view[written:]


class BankWriter:
    def __init__(self, path, *, n_samples, layers, provenance, max_write_bytes=256 * 1024 * 1024,
                 resume=False, guard=None):
        self.path = Path(path)
        self.n_samples = int(n_samples)
        self.layers = [int(layer) for layer in layers]
        if self.n_samples < 1 or not self.layers or len(set(self.layers)) != len(self.layers):
            raise BankError('A bank needs positive N and unique layer indices')
        if self.layers != sorted(self.layers):
            raise BankError('Layer order must be sorted, matching encoder intermediate outputs')
        self.provenance = json.loads(json.dumps(provenance))
        self.max_write_bytes = int(max_write_bytes)
        if self.max_write_bytes <= 0:
            raise BankError('max_write_bytes must be positive')
        self.guard = guard
        self._handles = {}
        self._hashes = {}
        self._failed = False
        self._closed = False
        self._progress_path = self.path / '.progress.json'
        self._lock_handle = None
        if self.path.exists():
            if not resume:
                raise BankError(f'Output exists; use a new path or explicit --resume-bank: {self.path}')
            if not self.path.is_dir() or (self.path / 'manifest.json').exists():
                raise BankError('Only an incomplete version-2 bank can be resumed')
        else:
            if resume:
                raise BankError('Cannot resume a bank directory that does not exist')
            self.path.mkdir(parents=True)
        # flock is released by the OS on exit/crash; leave the harmless lock file.
        import fcntl
        self._lock_handle = (self.path / '.writer.lock').open('a+b')
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock_handle.close()
            raise BankError('Another process owns this bank writer') from error
        try:
            if resume:
                self._state = json.loads(self._progress_path.read_text())
                if (self._state.get('version') != VERSION or self._state.get('n_samples') != self.n_samples
                        or self._state.get('layers') != self.layers or self._state.get('provenance') != self.provenance):
                    raise BankError('Resume requires exactly identical inputs, layers, N and provenance')
            else:
                self._state = {'version': VERSION, 'complete': False, 'n_samples': self.n_samples,
                               'layers': self.layers, 'provenance': self.provenance,
                               'written_samples': 0, 'files': {}, 'references_bytes': 0,
                               'references_sha256': hashlib.sha256().hexdigest(),
                               'reference_index_sha256': hashlib.sha256().hexdigest()}
                _atomic_json(self._progress_path, self._state)
            self.written_samples = int(self._state['written_samples'])
            if not 0 <= self.written_samples <= self.n_samples:
                raise BankError('Invalid committed sample count')
            self._restore_files(resume)
        except BaseException:
            self.close()
            raise

    def _restore_files(self, resume):
        for key, meta in self._state['files'].items():
            path = _safe_file(self.path, meta['file'])
            if path.stat().st_size != meta['bytes']:
                raise BankError(f'Incorrect preallocated bank size: {path}')
            prefix = self.written_samples * meta['row_bytes']
            digest = _hash_prefix(path, prefix, self.guard)
            if digest.hexdigest() != meta['committed_sha256']:
                raise BankError(f'Committed feature prefix was modified: {path}')
            handle = path.open('r+b', buffering=0)
            handle.seek(prefix)
            self._handles[key], self._hashes[key] = handle, digest
        for key, filename, length, hash_key in [
                ('references', 'references.jsonl', self._state['references_bytes'], 'references_sha256'),
                ('reference_index', 'references.idx', self.written_samples * 8, 'reference_index_sha256')]:
            path = _safe_file(self.path, filename)
            if resume and length and not path.exists():
                raise BankError(f'Missing committed references: {path}')
            handle = path.open('r+b' if path.exists() else 'w+b', buffering=0)
            self._handles[key] = handle
            digest = _hash_prefix(path, length, self.guard)
            if digest.hexdigest() != self._state[hash_key]:
                raise BankError('Committed reference records were modified')
            handle.truncate(length)
            handle.seek(length)
            self._hashes[key] = digest

    def _initialize_shapes(self, cls, patches):
        total_bytes = 0
        specs = {}
        for layer in self.layers:
            for kind, tensor in [('cls', cls[layer]), ('patches', patches[layer])]:
                shape = [self.n_samples] + list(tensor.shape[1:])
                row_bytes = math.prod(shape[1:]) * 4
                total_bytes += row_bytes * self.n_samples
                specs[f'{layer}:{kind}'] = {'file': f'layer_{layer}_{kind}.f32', 'shape': shape,
                                           'row_bytes': row_bytes, 'bytes': row_bytes * self.n_samples,
                                           'committed_sha256': hashlib.sha256().hexdigest()}
        if shutil.disk_usage(self.path).free < total_bytes + 64 * 1024 * 1024:
            raise BankError(f'Insufficient disk space for all float32 references ({total_bytes} bytes)')
        for key, meta in specs.items():
            handle = _safe_file(self.path, meta['file']).open('w+b', buffering=0)
            self._handles[key] = handle
            handle.truncate(meta['bytes'])
            os.fsync(handle.fileno())
            self._hashes[key] = hashlib.sha256()
        self._state['files'] = specs
        _atomic_json(self._progress_path, self._state)

    def append(self, cls_by_layer, patches_by_layer, references=None):
        """Commit one sequential batch. Tensors stay float32; no sample is dropped."""
        if self._closed or self._failed:
            raise BankError('Writer is closed or failed; reopen explicitly with resume=True')
        if set(cls_by_layer) != set(self.layers) or set(patches_by_layer) != set(self.layers):
            raise BankError('Each batch must contain every configured layer exactly once')
        batch = None
        for layer in self.layers:
            cls, patches = cls_by_layer[layer], patches_by_layer[layer]
            if (not isinstance(cls, torch.Tensor) or not isinstance(patches, torch.Tensor)
                    or cls.dtype != torch.float32 or patches.dtype != torch.float32):
                raise BankError('Feature tensors must be float32 (no implicit precision conversion)')
            if cls.ndim != 2 or patches.ndim != 3 or cls.shape[0] != patches.shape[0] or cls.shape[1] != patches.shape[2]:
                raise BankError('Expected CLS [B,C] and patches [B,L,C] with aligned dimensions')
            if batch is None:
                batch = cls.shape[0]
            if batch != cls.shape[0] or batch < 1 or min(cls.shape[1:]) < 1 or min(patches.shape[1:]) < 1:
                raise BankError('All layers must contain the same nonempty batch')
            for kind, tensor in [('cls', cls), ('patches', patches)]:
                if tensor.numel() * 4 > self.max_write_bytes:
                    raise BankError('One layer batch exceeds max_write_bytes; reduce extraction batch_size')
                meta = self._state['files'].get(f'{layer}:{kind}')
                if meta is not None and list(tensor.shape[1:]) != meta['shape'][1:]:
                    raise BankError('Feature shape changed between batches')
        if self.written_samples + batch > self.n_samples:
            raise BankError('Batch exceeds declared reference count')
        if references is None:
            references = ({'index': i} for i in range(self.written_samples, self.written_samples + batch))
        # Reference metadata is bounded by batch size, not N.
        refs = []
        reference_bytes = 0
        for i, record in enumerate(references):
            if i >= batch or record.get('index') != self.written_samples + i:
                raise BankError('Reference records must match the batch and original global indices')
            encoded = _json_bytes(record)
            if len(encoded) > IO_CHUNK:
                raise BankError('One reference record exceeds 1 MiB')
            reference_bytes += len(encoded)
            if reference_bytes > min(self.max_write_bytes, 16 * IO_CHUNK):
                raise BankError('Batch reference metadata exceeds its bounded buffer')
            refs.append(encoded)
        if len(refs) != batch:
            raise BankError('A reference record is required for each feature row')
        try:
            if self.guard is not None:
                self.guard.check()
            if not self._state['files']:
                self._initialize_shapes(cls_by_layer, patches_by_layer)
            for layer in self.layers:
                for kind, tensor in [('cls', cls_by_layer[layer]), ('patches', patches_by_layer[layer])]:
                    key = f'{layer}:{kind}'
                    # Only one layer's CPU copy exists at a time. No pinned/mapped cache.
                    cpu = tensor.detach().to(device='cpu', memory_format=torch.contiguous_format,
                                             copy=not tensor.is_contiguous())
                    array = cpu.numpy()
                    if sys.byteorder != 'little':
                        array = array.astype('<f4')
                    view = memoryview(array).cast('B')
                    _write_all(self._handles[key], view)
                    self._hashes[key].update(view)
                    del view, array, cpu
                    os.fsync(self._handles[key].fileno())
            for encoded in refs:
                offset = struct.pack('<Q', self._handles['references'].tell())
                _write_all(self._handles['reference_index'], offset)
                self._hashes['reference_index'].update(offset)
                _write_all(self._handles['references'], encoded)
                self._hashes['references'].update(encoded)
            for key in ['references', 'reference_index']:
                os.fsync(self._handles[key].fileno())
            next_count = self.written_samples + batch
            self._state['written_samples'] = next_count
            self._state['references_bytes'] = self._handles['references'].tell()
            self._state['references_sha256'] = self._hashes['references'].hexdigest()
            self._state['reference_index_sha256'] = self._hashes['reference_index'].hexdigest()
            for key, meta in self._state['files'].items():
                meta['committed_sha256'] = self._hashes[key].hexdigest()
            _atomic_json(self._progress_path, self._state)
            self.written_samples = next_count
            if self.guard is not None:
                self.guard.check()
        except BaseException:
            self._failed = True
            self.close()
            raise

    def finalize(self):
        if self._closed or self._failed or self.written_samples != self.n_samples:
            raise BankError('Cannot complete a closed, failed or partial bank')
        expected = self.provenance.get('references_sha256')
        if expected is not None and expected != self._state['references_sha256']:
            raise BankError('References changed during extraction; completed bank was not published')
        manifest = {'version': VERSION, 'complete': True, 'dtype': 'float32', 'byte_order': 'little',
                    'n_samples': self.n_samples, 'layers': self.layers, 'provenance': self.provenance,
                    'files': self._state['files'],
                    'references': {'file': 'references.jsonl', 'index_file': 'references.idx',
                                   'count': self.n_samples, 'bytes': self._state['references_bytes'],
                                   'sha256': self._state['references_sha256'],
                                   'index_sha256': self._state['reference_index_sha256']}}
        _atomic_json(self.path / 'manifest.json', manifest)
        self.close()
        return BankReader(self.path)

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        if self._lock_handle is not None:
            self._lock_handle.close()
            self._lock_handle = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class BankReader:
    def __init__(self, path, *, expected_provenance=None, max_read_bytes=DEFAULT_MAX_READ_BYTES):
        self.path = Path(path)
        if self.path.is_file():
            raise BankError('Legacy .pt/.pth banks are not implicitly loaded. Run python rad_bank.py migrate '
                            'with --legacy-memory-budget-gib and --provenance-json, or rebuild a streaming bank.')
        manifest_path = self.path / 'manifest.json'
        if not manifest_path.is_file():
            raise BankError('Bank is incomplete: manifest.json has not been committed; use --resume-bank to continue')
        self.manifest = json.loads(manifest_path.read_text())
        m = self.manifest
        if m.get('version') != VERSION or m.get('complete') is not True or m.get('dtype') != 'float32' or m.get('byte_order') != 'little':
            raise BankError('Unsupported or incomplete bank format')
        self.layers = m['layers']
        self.n_samples = self.N = int(m['n_samples'])
        self.provenance = m['provenance']
        if self.n_samples < 1 or not self.layers or self.layers != sorted(set(self.layers)):
            raise BankError('Invalid bank count or layer order')
        self.max_read_bytes = int(max_read_bytes)
        if self.max_read_bytes <= 0:
            raise BankError('max_read_bytes must be positive')
        if expected_provenance is not None:
            self.validate_provenance(expected_provenance)
        self.cls_shapes, self.patch_shapes = {}, {}
        for layer in self.layers:
            for kind, dims, shapes in [('cls', 2, self.cls_shapes), ('patches', 3, self.patch_shapes)]:
                meta = m['files'][f'{layer}:{kind}']
                shape = meta['shape']
                if (len(shape) != dims or shape[0] != self.n_samples or any(not isinstance(v, int) or v < 1 for v in shape)
                        or meta['row_bytes'] != math.prod(shape[1:]) * 4 or meta['bytes'] != math.prod(shape) * 4):
                    raise BankError('Invalid feature dimensions in manifest')
                if _safe_file(self.path, meta['file']).stat().st_size != meta['bytes']:
                    raise BankError('Truncated or oversized feature file')
                shapes[layer] = tuple(shape)
            if self.cls_shapes[layer][1] != self.patch_shapes[layer][2]:
                raise BankError('CLS and patch dimensions differ')
        self.layer_shapes = {layer: {'cls': self.cls_shapes[layer], 'patches': self.patch_shapes[layer]} for layer in self.layers}
        refs = m['references']
        if refs['count'] != self.n_samples:
            raise BankError('Reference count does not match feature rows')
        if (_safe_file(self.path, refs['file']).stat().st_size != refs['bytes']
                or _safe_file(self.path, refs['index_file']).stat().st_size != self.n_samples * 8):
            raise BankError('Reference sidecar is truncated')

    def __len__(self):
        return self.n_samples

    def validate_provenance(self, expected):
        _validate_expected(self.provenance, expected)

    def _read(self, layer, kind, indices, index_buffer_bytes=0):
        if layer not in self.layers:
            raise BankError(f'Layer {layer} is not present in bank')
        meta = self.manifest['files'][f'{layer}:{kind}']
        count = len(indices)
        # Peak is the returned array plus <= one MiB I/O buffer. Caller chunks
        # must fit this explicit cap; no hidden cache or lifetime mapping exists.
        output_bytes = count * meta['row_bytes']
        if output_bytes + min(IO_CHUNK, output_bytes) + index_buffer_bytes > self.max_read_bytes:
            raise BankError('Bank read exceeds max_read_bytes; reduce the caller reference/CLS chunk')
        if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)) or i < 0 or i >= self.n_samples for i in indices):
            raise IndexError('Reference index outside bank')
        array = np.empty((count, *meta['shape'][1:]), dtype='<f4')
        flat = memoryview(array).cast('B') if count else None
        fd = os.open(_safe_file(self.path, meta['file']), os.O_RDONLY)
        try:
            position = 0
            while position < count:
                end = position + 1
                while end < count and indices[end] == indices[end - 1] + 1:
                    end += 1
                remaining = (end - position) * meta['row_bytes']
                file_offset = int(indices[position]) * meta['row_bytes']
                destination = position * meta['row_bytes']
                while remaining:
                    chunk = os.pread(fd, min(IO_CHUNK, remaining), file_offset)
                    if not chunk:
                        raise BankError('Short read from feature file')
                    flat[destination:destination + len(chunk)] = chunk
                    remaining -= len(chunk)
                    destination += len(chunk)
                    file_offset += len(chunk)
                    del chunk
                position = end
        finally:
            os.close(fd)
        if sys.byteorder != 'little':
            array = array.astype(np.float32)
        return torch.from_numpy(array)

    def read_cls(self, layer, start, stop):
        if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start <= stop <= self.n_samples:
            raise IndexError('Invalid CLS slice')
        return self._read(layer, 'cls', range(start, stop))

    def read_patches(self, layer, indices):
        if layer not in self.layers:
            raise BankError(f'Layer {layer} is not present in bank')
        if not hasattr(indices, '__len__'):
            raise BankError('Provide a bounded, sized sequence of reference indices')
        row_bytes = self.manifest['files'][f'{layer}:patches']['row_bytes']
        if len(indices) * row_bytes > self.max_read_bytes:
            raise BankError('Bank read exceeds max_read_bytes; reduce the caller reference chunk')
        if isinstance(indices, torch.Tensor):
            if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
                raise BankError('Patch indices must be a one-dimensional integer tensor')
            # Avoid a Python int per index: for tiny feature rows that list can
            # dwarf the feature buffer. Account for a possible device-to-host
            # index copy before allocating it, and expose its NumPy view.
            index_bytes = indices.numel() * indices.element_size()
            output_bytes = len(indices) * row_bytes
            if output_bytes + min(IO_CHUNK, output_bytes) + index_bytes > self.max_read_bytes:
                raise BankError('Bank read exceeds max_read_bytes; reduce the caller reference chunk')
            indices = indices.detach().cpu().numpy()
            return self._read(layer, 'patches', indices, index_buffer_bytes=index_bytes)
        return self._read(layer, 'patches', indices)

    def iter_references(self):
        with _safe_file(self.path, self.manifest['references']['file']).open('rb') as handle:
            for index in range(self.n_samples):
                line = handle.readline(IO_CHUNK + 1)
                if not line or len(line) > IO_CHUNK:
                    raise BankError('Missing or oversized reference record')
                record = json.loads(line)
                if record.get('index') != index:
                    raise BankError('Reference ordering is inconsistent')
                yield record
            if handle.read(1):
                raise BankError('Unexpected extra reference records')

    def read_reference(self, index):
        if not isinstance(index, int) or not 0 <= index < self.n_samples:
            raise IndexError('Reference index outside bank')
        refs = self.manifest['references']
        fd = os.open(_safe_file(self.path, refs['index_file']), os.O_RDONLY)
        try:
            raw = os.pread(fd, 16 if index + 1 < self.n_samples else 8, index * 8)
        finally:
            os.close(fd)
        start = struct.unpack('<Q', raw[:8])[0]
        stop = struct.unpack('<Q', raw[8:])[0] if len(raw) == 16 else refs['bytes']
        if not 0 <= start < stop <= refs['bytes'] or stop - start > IO_CHUNK:
            raise BankError('Invalid reference index offsets')
        fd = os.open(_safe_file(self.path, refs['file']), os.O_RDONLY)
        try:
            record = json.loads(os.pread(fd, stop - start, start))
        finally:
            os.close(fd)
        if record.get('index') != index:
            raise BankError('Reference index mismatch')
        return record

    def verify_checksums(self, guard=None):
        """Optional sequential integrity verification, always a one-MiB buffer."""
        for meta in self.manifest['files'].values():
            if _hash_prefix(_safe_file(self.path, meta['file']), guard=guard).hexdigest() != meta['committed_sha256']:
                raise BankError('Feature file checksum mismatch')
        refs = self.manifest['references']
        for field, digest in [('file', 'sha256'), ('index_file', 'index_sha256')]:
            if _hash_prefix(_safe_file(self.path, refs[field]), guard=guard).hexdigest() != refs[digest]:
                raise BankError('Reference checksum mismatch')


def migrate_legacy(source, target, *, provenance, memory_budget_bytes, batch_size=8, references=None, guard=None):
    """Explicit small/medium legacy conversion under a preflighted resident budget.

    Legacy torch serialization cannot be independently read by row. This command
    intentionally refuses banks whose *uncompressed* storage and copies do not
    fit the supplied RAM budget. Rebuild those with the streaming builders. It
    never uses a persistent mmap as an unaccounted page-cache/RSS workaround.
    """
    source = Path(source)
    if not zipfile.is_zipfile(source):
        raise BankError('Legacy pre-ZIP checkpoints cannot be safely size-budgeted; rebuild a streaming bank')
    with zipfile.ZipFile(source) as archive:
        expanded = sum(item.file_size for item in archive.infolist())
    required = expanded * 3 + 64 * 1024 * 1024
    if required > memory_budget_bytes:
        raise BankError(f'Legacy conversion needs at least {required} budgeted RAM bytes; rebuild using the streaming builder')
    if guard is None:
        raise BankError('Legacy migration requires a resource guard with an explicit RAM budget')
    guard.preflight()
    guard.check(force=True)
    legacy = torch.load(source, map_location='cpu', weights_only=True)
    if not isinstance(legacy, dict) or not all(k in legacy for k in ('layers', 'cls_banks', 'patch_banks')):
        raise BankError('Unsupported legacy bank dictionary')
    layers, cls, patches = legacy['layers'], legacy['cls_banks'], legacy['patch_banks']
    if len(layers) != len(cls) or len(layers) != len(patches) or not layers:
        raise BankError('Legacy layers and tensor lists do not align')
    n = cls[0].shape[0]
    if any(t.shape[0] != n for t in cls + patches):
        raise BankError('Legacy tensor sample counts differ')
    if batch_size < 1:
        raise BankError('Migration batch_size must be positive')
    provenance = dict(provenance)
    provenance['legacy_migration'] = {'source_sha256': checkpoint_fingerprint(source), 'metadata_supplied_explicitly': True}
    iterator = iter(references) if references is not None else None
    with BankWriter(target, n_samples=n, layers=layers, provenance=provenance, guard=guard) as writer:
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            records = (next(iterator) for _ in range(stop - start)) if iterator is not None else None
            writer.append({layer: cls[i][start:stop] for i, layer in enumerate(layers)},
                          {layer: patches[i][start:stop] for i, layer in enumerate(layers)}, records)
        return writer.finalize()


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    migrate = sub.add_parser('migrate', help='Explicitly migrate a legacy bank only if its full load fits the RAM budget')
    migrate.add_argument('source')
    migrate.add_argument('target')
    migrate.add_argument('--legacy-memory-budget-gib', type=float, required=True)
    migrate.add_argument('--provenance-json', required=True, help='Verified encoder/preprocessing/category metadata; never infer it from filename')
    migrate.add_argument('--references-jsonl', required=True, help='Verified reference records in original bank order; legacy tensors contain no paths')
    migrate.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    from rad_resources import ResourceGuard
    budget = int(args.legacy_memory_budget_gib * 1024 ** 3)
    guard = ResourceGuard(ram_budget_bytes=budget, device='cpu')
    if args.references_jsonl:
        with open(args.references_jsonl) as handle:
            migrate_legacy(args.source, args.target, provenance=json.loads(Path(args.provenance_json).read_text()),
                           memory_budget_bytes=budget, batch_size=args.batch_size, references=(json.loads(line) for line in handle), guard=guard)
    else:
        migrate_legacy(args.source, args.target, provenance=json.loads(Path(args.provenance_json).read_text()),
                       memory_budget_bytes=budget, batch_size=args.batch_size, guard=guard)


if __name__ == '__main__':
    _main()
