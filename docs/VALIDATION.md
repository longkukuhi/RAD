# Validation of the bounded-memory revision

Validation date: 2026-10-19. Source base:
`59847f4f55082c31649d4e0a2dd0ec329b932633` on
`longkukuhi/RAD`; implementation branch:
`fix/bounded-memory-and-reproducibility`.
The tested source files and their SHA-256 values are recorded in
[validation/tested-source.json](validation/tested-source.json).

This revision was checked in an isolated local checkout. No real dataset
experiment, pretrained encoder load, GPU inference, extra scientific seed or
server deployment was performed. Existing experiment checkouts and results were
left untouched.

## Functional checks

**99 tests passed** in the final CPU run, using Python 3.12.14, torch 2.7.0,
torchvision 0.22.0, NumPy 2.2.6, SciPy 1.16.3, scikit-learn 1.6.1 and
scikit-image 0.25.2 on macOS 27.0.1 arm64. The full test process reached
394.09 MiB peak RSS (`resource.getrusage`, not a sample), under its 1 GiB
validation budget. Tests use small arrays, fake encoders or injected resource
snapshots. Lock tests also start a small standard-library child process.

Coverage includes:

- Float32 disk-bank round trips, corruption/incomplete-prefix rejection,
  bounded row reads, interrupted writing and verified recovery; mixed normal
  and anomalous training splits are rejected without filtering.
- Dense-versus-chunked patch scoring in global and positional modes, including
  radius 0 and oversized radii; layer fusion, query batches, image ties, budget
  refusal/adaptive chunks and reuse of shared reference reads. Float32 score
  comparisons use `rtol=1e-6, atol=2e-7` on these CPU fixtures.
- All seven metrics against their existing/reference formulas, including ties,
  multiple external-merge passes and input float16/32/64 preservation. AUPRO
  retains the original 200-step `arange` threshold grid, strict `>` comparison,
  connected-component convention and normalization by maximum retained FPR
  below 0.3. Fragmented masks are also covered.
- End-to-end synthetic evaluation and committed-prediction recovery; completed
  predictions do not load an encoder or start an empty DataLoader worker pool.
  Cached success respects explicit spool cleanup, and unrelated files remain.
- Exclusive metric-spool ownership across processes, refusal before truncation,
  exception/close-error lock release and persistent lock inode after cleanup.
- Resource rejection for insufficient RAM/VRAM, active paging, counter reset,
  missing host/shared-memory evidence, cgroup limits, and unresolved Windows
  paging bounds; short-window checks preserve the valid paging baseline.

`pip check` reports no broken requirements. All four entry-point `--help`
commands and the migration CLI import successfully. Static argparse validation
checked 10 README preparation/build/evaluation commands, including all eight
dataset recipes and matching category/path pairs. Scientific recipe options
were preserved while bank paths changed from `.pth` files to version-2
directories. See [CLI receipts](validation/cli-smoke.json) and
[README command audit](validation/readme-command-audit.json).

An initial SciPy 1.15.3 wheel failed to load its `_spropack` extension on this
macOS host (`__thread_bss` / missing TLV symbols). The final requirements use
1.16.3, which imported and passed the tests. This is a tested local compatibility
fix, not verification of all operating systems. The added GitHub Actions matrix
targets Linux Python 3.11/3.12; it has not yet run remotely.

Run the regression suite from the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m pip check
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
```

## Synthetic bank scaling

Three independent CPU processes used a 1 GiB RAM budget, resource admission,
the default system reserve, one numerical thread and synthetic seed 1. No model
was loaded. Each bank has two layers, 196 patches per image, 64 channels and
float32 features; build batches contain 16 references. Retrieval uses four
queries, `k_image=32`, positional radius 1, four references per read, 16 query
patches per chunk, 256 CLS rows per chunk and 16 MiB workspace/score caps.

| Reference images | Bank size on disk | Peak process RSS | Build time | Retrieval time |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 12.32 MiB | 193.89 MiB | 0.036 s | 0.164 s |
| 8,192 | 788.40 MiB | 191.17 MiB | 2.196 s | 0.199 s |
| 65,536 | 6.159 GiB | 196.27 MiB | 16.850 s | 0.185 s |

The largest bank is over six times the admitted RAM budget, yet the process did
not retain it in memory. The 512× increase in reference count changed measured
peak RSS by about 2.4 MiB relative to the smallest run. Small RSS variation
between processes is expected. Host RAM was 128 GiB; physical/available-memory
checks passed. Actual macOS swap-in and swap-out rates were zero in the recorded
admission/monitoring windows, including throughout the largest run.

The four queries intentionally share image neighbors: both layers together
read 64 reference rows, versus 256 without batch reuse. At most four reference
rows were read at once; all scores were finite. This is an I/O-count result,
not a claim of fourfold total speedup.

These short runs followed bank writing on a local SSD and can benefit from
warm OS caches. They do not measure cold storage, DINOv3 execution, default
768-channel/four-layer memory, metric disk throughput, network disks or CUDA.
RSS does not include OS page cache or other processes. A real experiment's peak
must include encoder, workers, preprocessing and configured evaluation buffers.

The [benchmark script](../tests/benchmark_streaming.py) is intentionally separate
from pytest. It performs real resource admission, writes a temporary bank, saves
receipts and deletes that synthetic bank afterward. From the repository root:

```bash
PYTHONPATH=. python tests/benchmark_streaming.py 128 /tmp/rad-memory-check
PYTHONPATH=. python tests/benchmark_streaming.py 8192 /tmp/rad-memory-check
PYTHONPATH=. python tests/benchmark_streaming.py 65536 /tmp/rad-memory-check
```

Run these sequentially only after checking the intended host's resources and
free disk space. The largest requires more than 6.2 GiB temporary storage.
Saved results are [128](validation/benchmark-128.json),
[8192](validation/benchmark-8192.json) and
[65536](validation/benchmark-65536.json); public receipts omit host process
inventories and local directory names.

## Remaining validation boundaries

- Full real-data/CUDA equivalence and throughput are unmeasured. Tiled kernels
  can change final-bit rounding and near-tie ordering. No paper benchmark score
  or universal peak/speed guarantee is claimed.
- Resource budgets are estimates and sampled safeguards, not OS-enforced
  allocation caps. A single kernel/allocation or a concurrent job can change
  pressure between checkpoints. Conservative headroom remains necessary.
- A read-only target-host probe found zero Linux swap deltas, but one elevated
  Windows disk-paging upper-bound interval. That neither proves actual pagefile
  activity nor establishes sustained paging. Host pagefile/shared-GPU evidence
  was insufficient to admit a real GPU run, so none was started.
- Original RAD code now has the requested strict noncommercial terms, including
  academic commercial collaborations. Upstream rights remain separate. The
  attribution gaps documented in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
  need rights-holder confirmation; adding notices does not establish ownership
  or finish legal clearance.

Operational limits, recovery and storage accounting are documented in
[MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md).
