# ROG validation: bounded-memory RAD

Validation date: 2026-10-19. The final tested implementation is **v4 (`76c15c6`)**.
Earlier v3 results and interrupted attempts are retained to explain the fixes.
All execution used an isolated checkout and the existing target-host runtime.
The [sanitized numerical receipt](validation/rog-validation.json) records the
measurements, source-receipt hashes and [tested source manifest](validation/tested-source.json).

## Environment and source versions

- Hardware: NVIDIA GeForce RTX 5090; approximately 31.84 GiB dedicated memory
  reported by the resource probe; NVIDIA driver 616.92.
- Runtime: Windows host with WSL2 Linux, Python 3.12.3, PyTorch 2.8.0+cu128,
  CUDA runtime 12.8, and cuDNN 9.10.2.
- Reported RAM totals: approximately 189.64 GiB on the Windows host and
  149.04 GiB exposed to Linux. These are the observed environment's capacities.
- Initial working v3 source revision: `6a7b090`.
- Final v4 source revision: `76c15c6`, which batches positional-index transfers
  within the bounded retrieval workspace. The 48-row bank was built by v3; v4
  changed only retrieval geometry construction/accounting and its regression test,
  so the same bank was reused after provenance validation.
- No packages were installed or changed on the target host. Its torch 2.8.0
  environment differs from the pinned local torch 2.7.0 environment.
- The comparison source is the original evaluation module at `59847f4`.
  Receipt/source checks are retained with the validation evidence.

Precision settings were recorded. Real-image runs used float32 features,
disabled TF32 matrix multiplication, and enabled deterministic cuDNN behavior.
The GPU synthetic test checked that its backend precision settings did not
change during the run. It did not silently enable reduced-precision arithmetic.

## Prespecified real-data protocol

The real-data check used the complete `1m1` category from 3D-ADAM: **48 normal
training references and 56 test images**, including 16 normal and 40 anomalous
test images. No image, reference, feature layer, or patch was sampled away.

| Setting | Value |
| --- | --- |
| Seed | `1`, selected before execution; no additional seeds |
| Encoder | DINOv3 ViT-B/16, same checkpoint for both evaluators |
| Preprocessing | Resize to 512, crop to 448 |
| Feature layers | `[3, 6, 9, 11]`, float32 |
| Batch / workers | `2` / `0` |
| Retrieval | `k_image=48`, positional bank enabled, radius `1` |
| Mask resolution / image score ratio | `256` / `0.01` |
| Layer weights / visualization | Default equal weighting / disabled |
| Registered task budgets | 8 GiB RAM and 8 GiB dedicated GPU memory |

The resource guard ran before model/bank loading and during execution. The
normal host reserve policy remained active: at least 8 GiB and 20% of RAM.

## Initial monitoring stops and corrections

Two earlier GPU attempts stopped before completing their comparison cases.
Their interruption receipts remain part of the evidence; they are not passes.

1. WSL/WDDM reported dedicated memory for a GPU process as `N/A`, causing
   `probe_failed`. The corrected probe uses whole-device `total - free` as an
   explicitly labelled conservative upper bound when exact process usage is
   unavailable. Unknown usage is never replaced with zero.
2. A fixed 64 MiB shared-memory growth rule caused `gpu_shared_growth` during
   CUDA initialization. Growth alone did not establish paging or exhausted
   RAM. The corrected guard accounts for the selected adapter's shared-memory
   total together with process RSS against the registered RAM budget, while
   continuing to check host/Linux capacity and paging evidence.

The probe also now resolves the actual visible CUDA UUID, maps it to a native
Windows LUID, and verifies that target adapter's counter on every check. Missing
or ambiguous mappings are rejected; the default collector supports single-node
`node_mask=1` / `phys_0` layouts.

These changes corrected measurement/accounting assumptions. They did not alter
the model, data split, seed, reference count, scientific precision, or metrics.

## Completed final correctness checks

The v4 CPU suite passed **130 tests plus 105 subtests**, with 22 source hashes
verified before execution. The JUnit aggregate count is 235 because it includes
subtests; it does not represent 235 top-level tests.
There were no failures, errors, or skips in that receipt.

Both v3 and v4 GPU comparisons passed **24 ordinary cases** and **3 exact-tie diagnostics**.
Ordinary cases checked scores, selected reference sets, float32 output, and
reference reuse under different chunk sizes. The largest score error was
`1.1920928955078125e-7`, within the prespecified `rtol=1e-5`, `atol=2e-6`.
All three tie probes preserved the reference set in this environment; this is
not a guarantee of identical tie ordering across every backend.

The real-image comparison passed all seven metrics at the prespecified absolute
tolerance `1e-5`. The maximum absolute difference was **`6.266705460111766e-8`**.

| Metric | Original function | v4 streaming evaluator | Absolute difference |
| --- | ---: | ---: | ---: |
| I-AUROC | 0.5734375000 | 0.5734375000 | 0 |
| I-AP | 0.8114263036 | 0.8114263036 | 0 |
| I-F1 | 0.8333332847 | 0.8333332847 | 0 |
| P-AUROC | 0.9444085877 | 0.9444085867 | 9.68425e-10 |
| P-AP | 0.3367015169 | 0.3367015796 | 6.26671e-8 |
| P-F1 | 0.4259332151 | 0.4259332151 | 0 |
| P-AUPRO | 0.8949227850 | 0.8949227681 | 1.69023e-8 |

## Memory, paging, and timing evidence

The following columns have different scopes. RSS is the main process's actual
`ru_maxrss` high-water mark. Allocated/reserved values are PyTorch CUDA allocator
peaks. The last column is a sampled **whole-device usage upper bound**, including
other users of that GPU; exact per-process GPU residency was unavailable.

| Completed run | RSS peak, MiB | CUDA allocated / reserved peak, MiB | Device-used upper bound, MiB |
| --- | ---: | ---: | ---: |
| v4 GPU synthetic comparison | 1484.58 | 17.89 / 28.00 | 1896 |
| Complete 48-reference bank build | 1281.98 | 404.73 / 464.00 | 2112 |
| v3 complete-category evaluation | 1873.64 | 415.52 / 466.00 | 2336 |
| v4 complete-category evaluation | 1867.67 | 415.52 / 466.00 | 2336 |
| Final original-function comparison | 2409.96 | 1044.73 / 1156.00 | 3026 |
| Completed-run CPU recovery | 612.62 | Not initialized | Not measured |

Guard-sampled RSS is separately recorded and can miss a process high-water
mark. Selected-adapter shared-memory totals can include other processes.
Their sum with main-process RSS is a conservative upper bound and can double
count resident pages; it is not an exact measurement of this task's RAM use.

All recorded valid Linux swap windows in the completed runs above and the
scaling run had zero input/output rates and zero endpoint counter deltas.
In final v4 GPU, real-data, recovery and scaling checks, the largest observed
Windows combined input/output upper bound was 0.0581 MiB/s, below the 1 MiB/s
threshold; none had an active high-rate window.
Recorded Linux PSI averages were zero, with unchanged cumulative stall counts.
Windows hard-page counters are an **upper bound** that can include file-backed
page-ins; they are not exact pagefile-only I/O measurements.
The v3 GPU synthetic run recorded one input upper-bound window of **4.36 MiB/s**
and no second consecutive active window. Other completed runs stayed below
1 MiB/s in their recorded windows. The guard's sustained-paging rule requires
two consecutive valid windows at combined input/output of at least 1 MiB/s.
Deferred windows were not treated as measured zeros. These receipts therefore
support neither an “all Windows paging was zero” nor an “exact pagefile I/O” claim.

The final scaling-v4 run wrote a **6.159 GiB** bank for 65,536 synthetic
references, two layers, 196 patches, and 64 float32 channels without loading
an encoder. Peak main-process RSS was **497.57 MiB**, within its 1 GiB budget.
Build time was 72.22 s and retrieval time 0.54 s; the largest reference read was
four rows. This tests bounded storage/retrieval at that synthetic shape.
OS file caches and other processes are outside RSS; this is not a claim that
the entire host used only 498 MiB. The bank was temporary and deleted afterward.
The earlier scaling-v2 run also completed (498.93 MiB RSS, 73.10 s build).

The real v3 CLI evaluation took **87.63 s**; the original-function harness took
**34.87 s** end to end. Their CLI, setup, guard-call, and metric-storage overheads
differ. These totals are not a controlled pure-algorithm speed ratio.

## Final performance and recovery

The same complete-category CLI took **29.16 s** after batching positional-index
transfers, compared with **87.63 s** before that change. Both used the same bank,
images, batch/chunk sizes, precision and protocol. The final original-function
comparison passed again in 32.50 s, with the same maximum metric difference.
These are single observed elapsed times on this host, not universal speedups;
OS caches, setup and monitoring overhead affect them.

Explicit `--resume-eval --cleanup-metric-spool` completed in **5.09 s** on CPU,
returned the same seven metrics, and did not initialize CUDA or reload the
encoder. Only the persistent `.metric.lock` inode remained in the spool after
cleanup. All validation children exited and the queue lock was available at
postflight. This tests recovery of a completed run; synthetic regression tests
cover interruption during prediction and bank writing.

One external validation launcher mistakenly substituted `dinov4` into the
encoder name while renaming its version label. The resource model rejected it
before model loading. The launcher was corrected, parameter assertions were
added, and a new output directory was used; the failed receipt was preserved.
This was not a RAD inference failure or an alternative model experiment.

## Limits

This is a complete-category smoke/equivalence check, **not a 23-category paper
reproduction**. Both evaluators consume the same newly built version-2 bank;
the baseline does not independently reproduce the original bank builder.
The original evaluation function uses the current equivalent `utils.compute_pro`
compatibility implementation because of the original pandas AUPRO API issue.
It is therefore not a recreation of the entire original dependency environment.
These measurements do not establish every dataset, encoder, machine, long-run
failure/recovery path, or larger real-image bank. See [memory management](MEMORY_MANAGEMENT.md)
and [general validation](VALIDATION.md) for implementation details and other checks.
