# Running RAD within a physical-memory budget

RAD stores the complete reference bank on disk and reads bounded feature blocks.
It retains every reference, float32 features, the selected layers, and the
original scoring protocol. This removes whole-bank residency and repeated
whole-bank copies; it does not make model execution or disk I/O free.

## Where memory goes

| Stage | Resident work | Grows on disk |
| --- | --- | --- |
| Extraction | Encoder, one query batch and selected output layers; bounded worker prefetch | All float32 CLS/patch rows and reference provenance |
| Image retrieval | CLS blocks, a scalar `batch_size × N` similarity matrix and top-k workspace | Complete CLS bank |
| Patch retrieval | One reference block, one query-patch block, normalized queries and running maxima | Complete patch bank |
| Prediction | One batch of resized maps/masks, convolution and image-score sorting buffers | Committed maps, masks, labels and scores |
| Metrics | Bounded external-sort/merge buffers and connected components of one image | Sorted runs and raw prediction spool |

No feature file is persistently memory-mapped. Explicit bounded reads avoid
turning a random-access full-bank mapping into an uncontrolled resident working
set. The operating system can still cache ordinary file reads and writes;
reclaimable file cache is different from anonymous tensors being swapped out.
Neither RSS alone nor the amount of already allocated swap proves active paging.

The scalar image-ranking matrix is deliberately retained to use the original
full `torch.topk` operation. It is **not constant in N**: float32 scores require
`4 × batch_size × N` bytes. `--max-score-mib` caps this matrix, and admission also
accounts for selection scratch and retained indices. Exceeding a cap raises an
error; it never drops references. Batched CLS matrix multiplication can differ
in last-bit rounding from an earlier kernel, so numerical tolerance checks do
not promise identical ranking for near ties on every device.

The same neighbor selected by several query images is read once per layer and
reference chunk. This reduces duplicate reads. It is most useful when queries
share neighbors; it is not a universal throughput multiplier.

## Admission and monitoring

Every builder and evaluator creates a resource guard before loading the encoder.
The guard checks physical/available RAM, a bounded process inventory, actual
local swap-counter deltas, effective Linux cgroup limits and dedicated NVIDIA
VRAM. CUDA under WSL additionally checks the Windows host and shared GPU memory.
Native Windows execution is not supported by the default probe or POSIX locks;
use Linux/WSL. CPU operation on macOS is supported.

Defaults:

- RAM reserve: `max(8 GiB, 20% of effective total RAM)`; WSL separately reserves
  the same rule against Windows-host RAM. Smaller machines need an explicitly
  adapted reserve, with room for the OS and other applications.
- Admission: two valid paging windows, approximately one second each.
- Monitoring: at cooperative batch/chunk checkpoints, normally every 5 seconds.
- Sustained paging: combined input/output at least 1 MiB/s in two valid windows.
  A used but inactive swap file alone does not cause rejection.
- WSL shared GPU usage: growth of at least 64 MiB from admission stops work for
  investigation. The counter covers Windows adapters; it does not identify
  which process caused growth.

`--ram-budget-gib` and `--gpu-budget-gib` mean **total conservative task peaks**,
not additional memory to allocate. The guard credits current main-process
residency once when calculating remaining demand. DataLoader workers belong in
the task estimate; only the main process is credited, so this calculation is
conservative. Logged sampled RSS is main-process RSS, not a measurement of the
whole worker tree or OS file cache.

The default analytic estimate includes encoder parameters, checkpoint loading,
float32 activations, an attention-matrix fallback, selected feature outputs,
source-image decoding, worker prefetch, retrieval/metric buffers, and batched
resized-map/convolution/sort buffers. A separate per-image allowance covers
connected-component metadata, including fragmented masks. These estimates are
not allocator-enforced limits or measured guarantees. Backend workspaces,
allocator caching and concurrent jobs vary; an indivisible operation can peak
between checks. Record a conservative measured budget on the target host before
large production runs, retaining the reserve. Do not lower a budget merely to
make admission pass.

Windows `PagesInputPersec`/`PagesOutputPersec` raw counters include mapped-file
hard faults, so they are logged as a **disk-paging upper bound**, not exact
pagefile traffic. A low bound verifies low pagefile traffic; a persistently high
bound is unresolved and stops work. It must not be attributed to RAD without
additional evidence. Missing/reset/stale counters also prevent continued work.
Forced checks shorter than a valid window still check capacity, record paging
as `deferred`, and retain the last valid rate baseline and consecutive count.

The guard raises an exception only in its own task. It does not kill another
job, change swap/pagefile settings, clear global caches, change WSL limits or
automatically restart an experiment.

## Tuning without changing the experiment

Start with the README's dataset recipe and keep its reference count, layers,
precision, preprocessing, `k_image`, positional radius and evaluation rules.
Tune execution controls explicitly and record them:

1. Reduce `--batch_size` and `--num_workers` to reduce encoder/decoder/prefetch
   residency. `prefetch_factor=1` and `pin_memory=False` are already used.
2. Reduce `--reference-chunk-size` and `--query-patch-chunk-size` when retrieval
   workspace is the limiting factor. The retriever also shrinks its internal
   chunks to fit `--memory-workspace-mib`. More chunks mean more loop/I/O cost.
3. Reduce `--cls-chunk-size` to bound CLS reads. The scalar ranking cap still
   applies independently; larger banks may require a smaller query batch.
4. Reduce `--metric-chunk-size` to trade sorting RAM for more disk merge work.
   Exact metrics still require a full-image mask for connected components.
5. Use a local SSD with enough free space for the bank and metric temporary
   files. A network or spinning disk can bottleneck exact retrieval even when
   RAM is low. Do not advertise a speedup until measured on the intended host.

Changing execution chunk sizes can change floating-point kernel reduction
details. The implementation retains the mathematical operations and float32
precision, with CPU tolerance tests. Any frozen scientific execution plan still
needs a new version and linkage to the original before execution changes.
All supplied entry points keep seed 1; no additional seeds are scheduled.

## Disk capacity

For N references, L layers, P patches and C channels, float32 storage is about
`4 × N × L × (P × C + C)` bytes, plus reference records and metadata. For the
four-layer ViT-B/16 example (P=784, C=768), patch rows alone cost 9,633,792 bytes
per reference image. Builders check capacity after learning actual feature
shapes, before writing the complete bank.

For M test images at H×W mask resolution, default float32 raw metric data costs
about `5 × M × H × W + 5 × M` bytes: one uint8 mask/label and one float32
map/score. Exact sorting additionally budgets up to two generations of
20-byte score/count records per image/pixel, plus block/filesystem overhead and
a 64 MiB reserve. At 10,000 images of 256×256 pixels, raw maps/masks are about
3.05 GiB and two worst-case pixel-sort generations about 24.4 GiB. Ties can
reduce actual sorted output, but capacity checks do not assume that reduction.

Disk demand is checked before prediction append and metric computation. An I/O
failure keeps the last committed prefix; uncommitted tails are truncated only
under the exclusive writer lock during explicit recovery. Atomic JSON receipts
and file flushes protect logical batch commits, but this is not a guarantee
against filesystem corruption or storage-device failure.

## Recovery

- For an interrupted builder, resolve the recorded resource/I/O cause and rerun
  the same command with `--resume-bank`. The writer verifies provenance and
  hashes of committed data before skipping those reference rows. A completed
  bank is opened by the evaluator, not overwritten by the builder.
- For interrupted evaluation, rerun with `--resume-eval`. Matching committed
  prediction batches are retained; incomplete metric sorting is rebuilt from
  the raw spool. When all requested predictions exist, metric recovery runs on
  CPU without loading the encoder or probing CUDA.
- `--cleanup-metric-spool` removes owned temporary prediction/sort files only
  after successful metrics publication, including a successful cached run on
  explicit resume. The persistent lock file/directory remain to prevent lock
  inode races. Files outside the spool's reserved data/sort paths are preserved;
  do not place personal files inside its generated sort directories.
- Do not launch two writers for the same bank or spool. Nonblocking process
  locks reject the second writer before truncating or appending anything.
- Use a new output name if data, checkpoint or scientific protocol changes.
  No stop schedules automatic recovery or an extra seed.

Inspect `resources.jsonl`, the category execution receipt and `interrupted.json`
for the cause and committed sample count. A system paging observation does not
prove that this task caused it. Preserve the evidence and reassess the host;
do not work around it by disabling swap or terminating unrelated jobs.

Legacy `.pth` migration necessarily loads its serialized tensors and therefore
requires an explicit conservative RAM budget (at least three times expanded
archive bytes plus 64 MiB), a successful resource check and externally verified
provenance/reference order. If that cannot fit, rebuild from normal images with
the streaming builder. The legacy file is preserved.

Measured implementation checks and their limits are in [VALIDATION.md](VALIDATION.md).
