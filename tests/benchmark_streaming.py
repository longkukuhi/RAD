"""Synthetic implementation benchmark; no model, dataset, or scientific run."""
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import torch
from rad_bank import BankReader, BankWriter
from rad_retrieval import ChunkedRetriever, RetrievalConfig
from rad_resources import ResourceGuard

torch.set_num_threads(1)
torch.manual_seed(1)
n = int(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
guard = ResourceGuard(1024**3, device='cpu', log_path=out / f'benchmark-{n}-resources.jsonl')
guard.preflight()
start = time.monotonic()
with tempfile.TemporaryDirectory(prefix=f'rad-benchmark-{n}-', dir=out) as temp:
    path = Path(temp) / 'bank'
    with BankWriter(path, n_samples=n, layers=[3,6], provenance={'synthetic_benchmark': True}, guard=guard) as writer:
        for first in range(0,n,16):
            guard.check()
            count = min(16,n-first)
            cls = {layer: torch.randn(count,64) for layer in [3,6]}
            patches = {layer: torch.randn(count,196,64) for layer in [3,6]}
            refs = [{'index': first+i, 'path': f'synthetic-{first+i}.png'} for i in range(count)]
            writer.append(cls, patches, refs)
        bank = writer.finalize()
    built = time.monotonic()
    bank_bytes = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    rows_read = 0
    original_read = bank.read_patches
    def counted(layer, ids):
        global rows_read
        rows_read += len(ids)
        return original_read(layer, ids)
    bank.read_patches = counted
    config = RetrievalConfig(4,16,256,16,16)
    retriever = ChunkedRetriever(bank,'cpu',config,guard)
    # All four queries intentionally share CLS neighbors to measure batch reuse.
    query_cls = [torch.randn(1,64).repeat(4,1) for _ in [3,6]]
    query_patches = [torch.randn(4,196,64) for _ in [3,6]]
    scores, ids = retriever.score_batch(query_patches,query_cls,k=32,positional=True,radius=1)
    guard.check(force=True)
    end = time.monotonic()
    result = {
        'synthetic': True, 'no_encoder_loaded': True, 'n_reference_images': n,
        'layers': 2, 'patches': 196, 'channels': 64, 'dtype': 'float32',
        'bank_bytes': bank_bytes, 'build_seconds': built-start, 'retrieval_seconds': end-built,
        'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'platform': sys.platform, 'ram_budget_bytes': 1024**3,
        'retrieval_config': vars(config), 'query_batch': 4, 'k_image': 32,
        'actual_reference_rows_read_across_layers': rows_read,
        'without_batch_reuse_rows': 4*32*2,
        'peak_reference_rows_per_read': retriever.peak_reference_rows,
        'scores_finite': bool(torch.isfinite(scores).all()),
        'score_shape': list(scores.shape), 'pid': os.getpid(),
        'scope': 'CPU synthetic storage/retrieval benchmark; no GPU or real dataset throughput claim',
    }
    if sys.platform != 'darwin':
        result['peak_process_rss_bytes'] *= 1024
    (out / f'benchmark-{n}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
