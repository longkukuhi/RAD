import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import rad_bank
from rad_bank import BankError, BankReader, BankWriter, migrate_legacy


LAYERS = [3, 6]
PROVENANCE = {'encoder': {'name': 'test_encoder', 'checkpoint_sha256': 'verified-test-sha'},
              'preprocessing': {'image_size': 512, 'crop_size': 448}, 'categories': ['a', 'b']}


def features(n=7):
    cls = {layer: torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) + layer for layer in LAYERS}
    patches = {layer: torch.arange(n * 3 * 4, dtype=torch.float32).reshape(n, 3, 4) + layer for layer in LAYERS}
    return cls, patches


def append(writer, cls, patches, start, stop):
    writer.append({layer: cls[layer][start:stop] for layer in LAYERS},
                  {layer: patches[layer][start:stop] for layer in LAYERS},
                  ({'index': i, 'path': f'/references/{i}.png', 'category': 'a' if i < 4 else 'b'}
                   for i in range(start, stop)))


def finished(path, n=7):
    cls, patches = features(n)
    with BankWriter(path, n_samples=n, layers=LAYERS, provenance=PROVENANCE) as writer:
        for start in range(0, n, 3):
            append(writer, cls, patches, start, min(n, start + 3))
        reader = writer.finalize()
    return reader, cls, patches


def test_roundtrip_preserves_every_float32_row_and_reference(tmp_path):
    reader, cls, patches = finished(tmp_path / 'bank')
    assert len(reader) == reader.N == 7
    assert reader.layers == LAYERS
    for layer in LAYERS:
        assert torch.equal(reader.read_cls(layer, 0, 7), cls[layer])
        assert torch.equal(reader.read_patches(layer, [6, 0, 4, 4, 1]), patches[layer][[6, 0, 4, 4, 1]])
        assert reader.read_patches(layer, []).shape == (0, 3, 4)
        assert reader.read_cls(layer, 2, 2).shape == (0, 4)
        assert reader.patch_shapes[layer] == (7, 3, 4)
    assert [r['index'] for r in reader.iter_references()] == list(range(7))
    assert reader.read_reference(4)['path'] == '/references/4.png'
    reader.verify_checksums()


def test_rejects_wrong_encoder_crop_category_order_and_bad_indices(tmp_path):
    reader, *_ = finished(tmp_path / 'bank')
    for expected in [{'encoder': {'checkpoint_sha256': 'other'}},
                     {'preprocessing': {'crop_size': 224}}, {'categories': ['b', 'a']}]:
        with pytest.raises(BankError, match='Mismatched'):
            BankReader(reader.path, expected_provenance=expected)
    BankReader(reader.path, expected_provenance={'preprocessing': {'crop_size': 448}})
    for indices in [[-1], [7], [1.5]]:
        with pytest.raises(IndexError):
            reader.read_patches(3, indices)
    with pytest.raises(IndexError):
        reader.read_cls(3, 0, 8)


def test_read_cap_checked_before_allocation_and_io(tmp_path, monkeypatch):
    reader, *_ = finished(tmp_path / 'bank')
    limited = BankReader(reader.path, max_read_bytes=100)
    def forbidden(*args, **kwargs):
        pytest.fail('over-budget read reached allocation or I/O')
    monkeypatch.setattr(rad_bank.np, 'empty', forbidden)
    monkeypatch.setattr(rad_bank.os, 'pread', forbidden)
    with pytest.raises(BankError, match='max_read_bytes'):
        limited.read_patches(3, [0, 1, 2])


def test_pread_is_block_bounded_and_not_per_patch(tmp_path, monkeypatch):
    cls = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    patches = torch.arange(2 * 70000 * 4, dtype=torch.float32).reshape(2, 70000, 4)
    with BankWriter(tmp_path / 'bank', n_samples=2, layers=[3], provenance={}) as writer:
        writer.append({3: cls}, {3: patches})
        reader = writer.finalize()
    real_pread = rad_bank.os.pread
    sizes = []
    def observed(fd, count, offset):
        sizes.append(count)
        return real_pread(fd, count, offset)
    monkeypatch.setattr(rad_bank.os, 'pread', observed)
    assert torch.equal(reader.read_patches(3, [0, 1]), patches)
    assert max(sizes) <= rad_bank.IO_CHUNK
    assert len(sizes) == 3  # 2.24 MB in three blocks, not 140,000 patch reads.


def test_tensor_indices_keep_a_bounded_numeric_buffer(tmp_path, monkeypatch):
    reader, _, patches = finished(tmp_path / 'bank')
    indices = torch.tensor([6, 0, 4, 4, 1], dtype=torch.int64)
    expected = patches[3][indices]
    def forbidden(*args, **kwargs):
        pytest.fail('tensor indices were expanded into a Python integer list')
    monkeypatch.setattr(torch.Tensor, 'tolist', forbidden)
    assert torch.equal(reader.read_patches(3, indices), expected)
    limited = BankReader(reader.path, max_read_bytes=500)  # 240 output + 240 I/O + 40 index > 500.
    with pytest.raises(BankError, match='max_read_bytes'):
        limited.read_patches(3, indices)


def test_noncontiguous_features_write_in_one_contiguous_conversion(tmp_path, monkeypatch):
    cls = torch.arange(8, dtype=torch.float32).reshape(4, 2).t()
    patches = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).transpose(1, 2)
    assert not cls.is_contiguous() and not patches.is_contiguous()
    original = torch.Tensor.to
    conversions = []
    def observed(tensor, *args, **kwargs):
        if kwargs.get('device') == 'cpu':
            conversions.append(kwargs.get('memory_format'))
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', observed)
    with BankWriter(tmp_path / 'bank', n_samples=2, layers=[3], provenance={}) as writer:
        writer.append({3: cls}, {3: patches})
        reader = writer.finalize()
    assert conversions == [torch.contiguous_format, torch.contiguous_format]
    assert torch.equal(reader.read_cls(3, 0, 2), cls)
    assert torch.equal(reader.read_patches(3, [0, 1]), patches)


def test_partial_resume_rolls_back_uncommitted_tails(tmp_path):
    path = tmp_path / 'bank'
    cls, patches = features()
    with BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE) as writer:
        append(writer, cls, patches, 0, 3)
    with pytest.raises(BankError, match='incomplete'):
        BankReader(path)
    # Simulate a process killed after raw writes but before its progress commit.
    with (path / 'layer_3_patches.f32').open('r+b') as handle:
        handle.seek(3 * 3 * 4 * 4)
        handle.write(b'bad uncommitted bytes')
    with (path / 'references.jsonl').open('ab') as handle:
        handle.write(b'bad uncommitted reference')
    with (path / 'references.idx').open('ab') as handle:
        handle.write(b'bad tail')
    with BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE, resume=True) as writer:
        assert writer.written_samples == 3
        append(writer, cls, patches, 3, 7)
        reader = writer.finalize()
    assert torch.equal(reader.read_patches(3, list(range(7))), patches[3])
    reader.verify_checksums()


def test_resume_refuses_changed_inputs_and_committed_corruption(tmp_path):
    path = tmp_path / 'bank'
    cls, patches = features()
    with BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE) as writer:
        append(writer, cls, patches, 0, 3)
    with pytest.raises(BankError, match='identical'):
        BankWriter(path, n_samples=7, layers=LAYERS, provenance={'changed': True}, resume=True)
    with (path / 'layer_3_cls.f32').open('r+b') as handle:
        handle.write(b'xxxx')
    with pytest.raises(BankError, match='modified'):
        BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE, resume=True)


def test_failed_commit_does_not_publish_or_advance_progress(tmp_path, monkeypatch):
    path = tmp_path / 'bank'
    cls, patches = features()
    writer = BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE)
    real_atomic = rad_bank._atomic_json
    def interrupted(path, value):
        if value.get('written_samples', 0) > 0:
            raise OSError('simulated interruption before progress publish')
        real_atomic(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(rad_bank, '_atomic_json', interrupted)
        with pytest.raises(OSError, match='simulated'):
            append(writer, cls, patches, 0, 3)
    assert json.loads((path / '.progress.json').read_text())['written_samples'] == 0
    assert not (path / 'manifest.json').exists()
    with BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE, resume=True) as writer:
        append(writer, cls, patches, 0, 7)
        reader = writer.finalize()
    reader.verify_checksums()


def test_guard_stop_preserves_last_atomic_batch(tmp_path):
    class Guard:
        calls = 0
        def check(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('paging stop')
    path = tmp_path / 'bank'
    cls, patches = features()
    with BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE, guard=Guard()) as writer:
        with pytest.raises(RuntimeError, match='paging stop'):
            append(writer, cls, patches, 0, 3)
    assert json.loads((path / '.progress.json').read_text())['written_samples'] == 3
    with BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE, resume=True) as writer:
        assert writer.written_samples == 3
        append(writer, cls, patches, 3, 7)
        writer.finalize().verify_checksums()


def test_rejects_half_precision_and_incomplete_finalization(tmp_path):
    cls, patches = features()
    with BankWriter(tmp_path / 'bank', n_samples=7, layers=LAYERS, provenance=PROVENANCE) as writer:
        with pytest.raises(BankError, match='float32'):
            writer.append({layer: t.half() for layer, t in cls.items()}, patches)
        append(writer, cls, patches, 0, 3)
        with pytest.raises(BankError, match='partial'):
            writer.finalize()


def test_reference_fingerprint_mismatch_cannot_publish_bank(tmp_path):
    cls, patches = features()
    provenance = dict(PROVENANCE, references_sha256='a different input fingerprint')
    with BankWriter(tmp_path / 'bank', n_samples=7, layers=LAYERS, provenance=provenance) as writer:
        append(writer, cls, patches, 0, 7)
        with pytest.raises(BankError, match='changed during extraction'):
            writer.finalize()
    assert not (tmp_path / 'bank' / 'manifest.json').exists()


def test_one_writer_lock_and_no_overwrite(tmp_path):
    path = tmp_path / 'bank'
    with BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE):
        with pytest.raises(BankError, match='Another process'):
            BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE, resume=True)
        with pytest.raises(BankError, match='Output exists'):
            BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE)


def test_legacy_never_implicitly_loads_or_bypasses_budget(tmp_path, monkeypatch):
    cls, patches = features()
    source = tmp_path / 'legacy.pth'
    torch.save({'layers': LAYERS, 'cls_banks': list(cls.values()), 'patch_banks': list(patches.values())}, source)
    def forbidden(*args, **kwargs):
        pytest.fail('legacy checkpoint was loaded without fitting its explicit budget')
    monkeypatch.setattr(rad_bank.torch, 'load', forbidden)
    with pytest.raises(BankError, match='not implicitly loaded'):
        BankReader(source)
    with pytest.raises(BankError, match='budgeted RAM'):
        migrate_legacy(source, tmp_path / 'new', provenance=PROVENANCE, memory_budget_bytes=1)


def test_explicit_legacy_migration_preserves_values(tmp_path):
    cls, patches = features()
    source = tmp_path / 'legacy.pth'
    torch.save({'layers': LAYERS, 'cls_banks': list(cls.values()), 'patch_banks': list(patches.values())}, source)
    guard = SimpleNamespace(preflight=lambda: None, check=lambda **kwargs: None)
    reader = migrate_legacy(source, tmp_path / 'new', provenance=PROVENANCE,
                            memory_budget_bytes=128 * 1024 ** 2, batch_size=3, guard=guard,
                            references=({'index': i, 'path': f'/ref/{i}.png'} for i in range(7)))
    assert torch.equal(reader.read_patches(3, [0, 1, 6]), patches[3][[0, 1, 6]])
    assert reader.provenance['legacy_migration']['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_builder_streams_in_batches_and_resumes_without_reencoding(tmp_path):
    # Extract the actual builder function without importing vision/backbone
    # dependencies or loading a model. Its CPU synthetic encoder is only 4-D.
    source = Path(rad_bank.__file__).with_name('build_bank_multilayer.py')
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_memory_bank_multilayer')
    namespace = {'torch': torch, 'BankWriter': BankWriter,
                 'DataLoader': torch.utils.data.DataLoader, 'Subset': torch.utils.data.Subset}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    encoded = []
    class Encoder:
        def eval(self):
            return self
        def get_intermediate_layers(self, images, **kwargs):
            encoded.extend(images[:, 0].tolist())
            return [(images[:, None, :].repeat(1, 3, 1) + layer, images + layer) for layer in kwargs['n']]
    cls, patches = features()
    images = torch.arange(7 * 4, dtype=torch.float32).reshape(7, 4)
    dataset = torch.utils.data.TensorDataset(images, torch.zeros(7), torch.arange(7))
    path = tmp_path / 'bank'
    with BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE) as writer:
        # Construct the exact encoder results for the initial committed rows.
        writer.append({layer: (images + layer)[:3] for layer in LAYERS},
                      {layer: (images[:, None, :].repeat(1, 3, 1) + layer)[:3] for layer in LAYERS})
    guard = SimpleNamespace(check=lambda **kwargs: None)
    with BankWriter(path, n_samples=7, layers=LAYERS, provenance=PROVENANCE, resume=True) as writer:
        reader = namespace['build_memory_bank_multilayer'](Encoder(), dataset, 'cpu', batch_size=2,
            layer_idx_list=LAYERS, num_workers=0, writer=writer,
            reference_for_index=lambda i: {'index': i}, guard=guard)
    assert encoded == [12.0, 16.0, 20.0, 24.0]
    assert torch.equal(reader.read_cls(3, 0, 7), images + 3)
    assert torch.equal(reader.read_patches(6, [0, 3, 6]), (images[:, None, :].repeat(1, 3, 1) + 6)[[0, 3, 6]])


def training_split_validator():
    source = Path(rad_bank.__file__).with_name('build_bank_multilayer.py')
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_normal_training_split')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[node.name]


def test_normal_imagefolder_rejects_extra_or_wrong_classes_without_filtering():
    validate = training_split_validator()
    validate(SimpleNamespace(classes=['good']), 'carpet')
    for classes in [['good', 'crack'], ['normal'], ['crack']]:
        dataset = SimpleNamespace(classes=list(classes), samples=[('kept.png', 0)])
        with pytest.raises(ValueError, match='only the good class'):
            validate(dataset, 'carpet')
        assert dataset.classes == classes
        assert dataset.samples == [('kept.png', 0)]


def test_normal_realiad_rejects_anomalous_train_without_filtering():
    validate = training_split_validator()
    validate(SimpleNamespace(labels=[False, 0, False]), 'audiojack', realiad=True)
    for labels in [[False, True], [0, 1], [0, -1]]:
        dataset = SimpleNamespace(labels=list(labels))
        with pytest.raises(ValueError, match='train contains anomalous labels'):
            validate(dataset, 'audiojack', realiad=True)
        assert dataset.labels == labels
