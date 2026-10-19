# Third-party notices and source inventory

This repository contains third-party code and references to separately licensed
datasets and model weights. The repository's original-code license does not
replace, restrict, or relicense third-party material. In mixed-source files, the
upstream material retains its own terms. Preserve existing copyright and license
headers when redistributing it.

This inventory was prepared on 2026-10-19 from the local source headers, README
links, and official upstream repositories. It identifies known notices and
remaining attribution gaps; it is not a completed legal clearance or a statement
that all material in the repository is available under one license.

## Identified material

| Local material | Evidence and attribution | License documents |
| --- | --- | --- |
| `dinov1/` and the DINO-derived portions of `models/vision_transformer.py` | Source headers credit Facebook, Inc. and affiliates and state Apache License 2.0. Both vision-transformer files also identify DINO and timm as sources. | [DINO Apache 2.0](licenses/DINO-APACHE-2.0.txt), [timm Apache 2.0](licenses/TIMM-APACHE-2.0.txt) |
| `dinov2/` | The 131 local Python source files inspected carry Meta Platforms, Inc. and affiliates / Apache License 2.0 headers. | [DINOv2 Apache 2.0](licenses/DINOv2-APACHE-2.0.txt) |
| `dinov3/` | The 28 local Python source files inspected credit Meta Platforms, Inc. and affiliates and refer to the DINOv3 License Agreement. | [DINOv3 License Agreement](licenses/DINOv3-LICENSE.md) |
| `beit/vision_transformer.py` | The file credits Microsoft (2021), contains an MIT license statement, credits modifications to Ross Wightman (2021), and names BEiT/BEiT2, timm, DeiT and DINO as sources. The file-level MIT statement must be retained; the timm project's general Apache license is not a reason to overwrite it. | [Microsoft UniLM MIT](licenses/BEiT-UNILM-MIT.txt), [UniLM upstream notices](licenses/UNILM-NOTICE.md), [timm Apache 2.0](licenses/TIMM-APACHE-2.0.txt), [DeiT Apache 2.0](licenses/DeiT-APACHE-2.0.txt), [DINO Apache 2.0](licenses/DINO-APACHE-2.0.txt) |
| `models/vit_encoder.py`, specifically `download_cached_file` | Its docstring explicitly says it is based on timm and cites upstream commit `29fda20e6d428bf636090ab207bbcf60617570ca`, `timm/models/_hub.py`. This is evidence for the identified helper, not a license determination for every other function in the file. | [timm Apache 2.0 at the cited commit](licenses/TIMM-APACHE-2.0.txt) |
| `prepare_data/prepare_visa.py` | Existing Amazon.com, Inc. or affiliates copyright and `SPDX-License-Identifier: Apache-2.0` header; the README identifies Amazon's SPot-the-Difference repository as the preparation source. | [SPot-the-Difference code Apache 2.0](licenses/VisA-SPOT-DIFF-APACHE-2.0.txt), [Amazon notice](licenses/VisA-SPOT-DIFF-NOTICE.txt) |
| VisA dataset references and `prepare_data/split_csv/` | The README points to Amazon's VisA distribution and its official splits. The upstream notice distinguishes Apache-licensed utility code from the dataset's Creative Commons Attribution 4.0 terms. Dataset images are not supplied by these license copies. The historical import and precise file-level scope of the local split CSVs still need confirmation. | [Amazon notice](licenses/VisA-SPOT-DIFF-NOTICE.txt), [VisA dataset CC BY 4.0](licenses/VisA-DATASET-LICENSE.txt) |

The UniLM notice is an unmodified notice from the complete upstream repository;
its list of components is not an assertion that all those components are vendored
here. Likewise, the current DINOv2 repository contains additional licenses for
separate Cell-DINO/X-Ray-DINO materials. This inventory does not assign those
licenses to this checkout's DINOv2 files merely because they share an upstream
repository name.

## Pinned official sources

Every file below was retrieved as raw content through the official repository's
GitHub API at the fixed commit shown. The files in `licenses/` preserve upstream
bytes, including upstream copyright notices and line endings.

| Upstream repository | Fixed license-source commit | Official source |
| --- | --- | --- |
| Facebook Research DINO | `7c446df5b9f45747937fb0d72314eb9f7b66930a` | [LICENSE](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/LICENSE) |
| Meta DINOv2 | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | [LICENSE](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE) |
| Meta DINOv3 | `6876159a11b4df116f30f667f8c9888617df0751` | [LICENSE.md](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md) |
| Microsoft UniLM / BEiT | `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd` | [LICENSE](https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/LICENSE), [NOTICE.md](https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/NOTICE.md) |
| Hugging Face / Ross Wightman timm | `29fda20e6d428bf636090ab207bbcf60617570ca` | [LICENSE](https://github.com/huggingface/pytorch-image-models/blob/29fda20e6d428bf636090ab207bbcf60617570ca/LICENSE) |
| Facebook Research DeiT | `7e160fe43f0252d17191b71cbb5826254114ea5b` | [LICENSE](https://github.com/facebookresearch/deit/blob/7e160fe43f0252d17191b71cbb5826254114ea5b/LICENSE) |
| Amazon SPot-the-Difference / VisA | `2a692ab575001cbde74d402d897a7286086c6199` | [LICENSE](https://github.com/amazon-science/spot-diff/blob/2a692ab575001cbde74d402d897a7286086c6199/LICENSE), [NOTICE](https://github.com/amazon-science/spot-diff/blob/2a692ab575001cbde74d402d897a7286086c6199/NOTICE), [LICENSE-DATASET](https://github.com/amazon-science/spot-diff/blob/2a692ab575001cbde74d402d897a7286086c6199/LICENSE-DATASET) |

[licenses/SOURCES.json](licenses/SOURCES.json) records the complete upstream URL,
commit, Git blob ID, SHA-256 and byte count for each saved document. Except for the
timm helper's explicitly cited source revision, these pinned commits identify
where license documents were obtained, **not** when the local code was copied.
Even for that helper, the precise local modification/import history has not been
reconstructed. Historical vendoring commits remain unverified.

## Attribution and scope still requiring confirmation

- `models/resnet.py` and `models/de_resnet.py` have no explicit license or copyright
  header. Familiar architecture names, coding style and model-download URLs do
  not establish authorship, permission or a particular license. Their origins and
  any required notices must be confirmed with the repository author or rights
  holder before treating them as original or relicensing them.
- The full extent and modification history of copied code in mixed-origin files,
  including the BEiT and transformer implementations, has not been established.
  Adding these notices does not supply missing rights or resolve conflicting
  grants. Other files without explicit attribution are not certified as wholly
  original by omission from the table.
- Dataset assets, pretrained checkpoints and installed Python dependencies retain
  their own applicable terms. A source-code license does not automatically grant
  rights to training data or weights. This change neither downloads nor bundles
  those assets, and is not an exhaustive dependency-license inventory.

No source-file header was replaced as part of this notice restoration. The
repository owner's original-code restrictions apply only within the scope of
rights the owner actually holds; they do not impose an additional noncommercial
restriction on independently licensed upstream material.
