"""Exact, float32 RAD retrieval with bounded feature and similarity buffers.

All references and selected patches participate. No quantization, subsampling,
approximate index, reduced precision, or changed positional neighborhood is used.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RetrievalConfig:
    reference_chunk_size: int = 8
    query_patch_chunk_size: int = 32
    cls_chunk_size: int = 4096
    workspace_mib: int = 256
    max_score_mib: int = 128

    def validate(self):
        for name, value in vars(self).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')


class ChunkedRetriever:
    def __init__(self, bank, device, config=None, guard=None):
        self.bank = bank
        self.device = torch.device(device)
        self.config = config or RetrievalConfig()
        self.config.validate()
        self.guard = guard
        self.peak_reference_rows = 0
        self.peak_similarity_elements = 0

    def _check(self):
        if self.guard is not None:
            self.guard.check()

    @torch.no_grad()
    def image_neighbors(self, query_cls, k):
        """Keep only scalar image scores on device, never all CLS features.

        A full scalar score matrix preserves torch.topk's existing tie selection
        instead of introducing a new merge tie rule. Its allocation has a hard
        cap; one million references need about 4 MiB per query, not feature-sized
        buffers. CLS blocks are read once and reused across the query batch.
        """
        n = self.bank.n_samples
        if not 1 <= k <= n:
            raise ValueError(f'k_image must be between 1 and {n}, got {k}')
        if query_cls.ndim != 2 or query_cls.dtype != torch.float32:
            raise ValueError('Expected float32 CLS features [batch, channels]')
        batch = len(query_cls)
        score_bytes = batch * n * query_cls.element_size()
        if score_bytes > self.config.max_score_mib * 2**20:
            raise MemoryError(
                f'Image-ranking scores need {score_bytes / 2**20:.1f} MiB; '
                'increase --max-score-mib and the declared RAM/VRAM budget. '
                'No reference images have been dropped.')
        # Include the retained B*k index tensor, topk values/indices, and a
        # selection scratch. CPU topk uses an aligned (value,int64) array, so a
        # scalar-vector-only estimate is insufficient; allow 32 bytes per score.
        ranking_bytes = batch * k * (8 * 5 + 12) + score_bytes * 9
        if ranking_bytes > (self.config.workspace_mib + self.config.max_score_mib) * 2**20:
            raise MemoryError('Image ranking and retained top-k indices exceed the declared buffers')
        layer = self.bank.layers[-1]
        channels = query_cls.shape[-1]
        workspace = self.config.workspace_mib * 2**20
        # Host read buffer, transfer, normalized copy, and normalization scratch.
        bytes_per_row = channels * 4 * 4 + batch * 4
        rows = min(self.config.cls_chunk_size, max(1, workspace // bytes_per_row))
        if bytes_per_row > workspace:
            raise MemoryError('Workspace cannot hold one CLS feature')
        query_cls = F.normalize(query_cls.to(self.device), dim=-1)
        similarities = torch.empty((batch, n), device=self.device, dtype=query_cls.dtype)
        for start in range(0, n, rows):
            self._check()
            stop = min(start + rows, n)
            cls = self.bank.read_cls(layer, start, stop).to(self.device)
            cls = F.normalize(cls, dim=-1)
            similarities[:, start:stop] = torch.matmul(query_cls, cls.t())
            del cls
        if not torch.isfinite(similarities).all():
            raise ValueError('Non-finite CLS similarity; check checkpoint and bank')
        values, neighbors = torch.topk(similarities, k, dim=-1)
        del similarities, values
        return neighbors

    @staticmethod
    def _positions(start, stop, grid, radius, device):
        positions = []
        for j in range(start, stop):
            row, col = divmod(j, grid)
            positions.append([
                r * grid + c
                for r in range(max(0, row-radius), min(grid, row+radius+1))
                for c in range(max(0, col-radius), min(grid, col+radius+1))
            ])
        width = max(map(len, positions))
        indices = torch.zeros((stop-start, width), dtype=torch.long, device=device)
        valid = torch.zeros((stop-start, width), dtype=torch.bool, device=device)
        for row, ids in enumerate(positions):
            indices[row, :len(ids)] = torch.tensor(ids, device=device)
            valid[row, :len(ids)] = True
        return indices, valid

    def _chunks(self, patches, channels, positional, radius, batch=1, reserved_bytes=0):
        cfg = self.config
        q = min(cfg.query_patch_chunk_size, patches)
        k = cfg.reference_chunk_size
        positions = min((2*radius+1)**2, patches) if positional else patches
        budget = cfg.workspace_mib * 2**20
        def estimate(qsize, ksize):
            # Include host/raw/normalized feature buffers, positional gather and
            # a possible einsum layout copy. Global mode needs only similarities.
            features = ksize * patches * channels * 4 * 4
            if positional:
                work = ksize * qsize * positions * channels * 4 * 2
            else:
                work = qsize * ksize * patches * 4 * 2
            return (features + work + patches * channels * 4 * (batch + 1)
                    + batch * patches * 4 * 4 + reserved_bytes)
        while estimate(q, k) > budget and (k > 1 or q > 1):
            if k > 1:
                k = max(1, k // 2)
            else:
                q = max(1, q // 2)
        if estimate(q, k) > budget:
            raise MemoryError('Workspace too small for one reference image and query patch')
        return q, k

    @torch.no_grad()
    def patch_scores(self, layer, query, neighbor_indices, positional=False, radius=1):
        if query.ndim != 2 or query.dtype != torch.float32:
            raise ValueError('Expected float32 patch features [patches, channels]')
        if radius < 0:
            raise ValueError('pos_radius must be nonnegative')
        length, channels = query.shape
        grid = math.isqrt(length)
        if grid * grid != length:
            raise ValueError('RAD expects a square patch grid')
        if tuple(self.bank.patch_shapes[layer][1:]) != (length, channels):
            raise ValueError('Bank/query patch grid or feature dimension mismatch')
        query = F.normalize(query.to(self.device), dim=-1)
        if neighbor_indices.ndim != 1 or neighbor_indices.numel() == 0:
            raise ValueError('At least one reference image is required')
        qchunk, rchunk = self._chunks(length, channels, positional, radius)
        best = torch.full((length,), -torch.inf, dtype=query.dtype, device=self.device)
        for first in range(0, len(neighbor_indices), rchunk):
            self._check()
            selected = neighbor_indices[first:first+rchunk].detach().cpu().tolist()
            self.peak_reference_rows = max(self.peak_reference_rows, len(selected))
            raw = self.bank.read_patches(layer, selected)
            neighbors = raw.to(self.device)
            neighbors = F.normalize(neighbors, dim=-1)
            del raw
            for start in range(0, length, qchunk):
                self._check()
                stop = min(start + qchunk, length)
                if positional:
                    pos, valid = self._positions(start, stop, grid, radius, self.device)
                    local = neighbors[:, pos, :]
                    sims = torch.einsum('qc,kqpc->qkp', query[start:stop], local)
                    sims.masked_fill_(~valid[:, None, :], -torch.inf)
                    maximum = sims.flatten(1).max(dim=-1).values
                    del local, pos, valid
                else:
                    sims = torch.matmul(query[start:stop], neighbors.reshape(-1, channels).t())
                    maximum = sims.max(dim=-1).values
                self.peak_similarity_elements = max(self.peak_similarity_elements, sims.numel())
                best[start:stop] = torch.maximum(best[start:stop], maximum)
                del sims, maximum
            del neighbors
        if not torch.isfinite(best).all():
            raise ValueError('Non-finite patch scores; check checkpoint and bank')
        return 1.0 - best

    @torch.no_grad()
    def score_batch(self, patch_features, cls_features, k, positional=False, radius=1, weights=None):
        layers = self.bank.layers
        if len(patch_features) != len(layers) or len(cls_features) != len(layers):
            raise ValueError('Feature layer count differs from the bank')
        if weights is None:
            weights = [1.0 / len(layers)] * len(layers)
        else:
            if len(weights) != len(layers) or not all(math.isfinite(w) for w in weights):
                raise ValueError('Provide one finite layer weight per bank layer')
            total = sum(weights)
            weights = ([w / total for w in weights] if total > 0
                       else [1.0 / len(layers)] * len(layers))
        topk = self.image_neighbors(cls_features[-1], k)
        sorted_ids = topk.detach().cpu().sort(dim=-1).values
        unique_ids = torch.unique(sorted_ids, sorted=True)
        batch, length, channels = patch_features[0].shape
        grid = math.isqrt(length)
        if grid * grid != length or radius < 0:
            raise ValueError('Expected a square grid and nonnegative positional radius')
        qchunk, rchunk = self._chunks(length, channels, positional, radius,
                                      batch=batch, reserved_bytes=batch * k * 8 * 5)
        fused = torch.zeros((batch, length), dtype=torch.float32, device=self.device)
        for li, layer in enumerate(layers):
            feature = patch_features[li]
            if feature.dtype != torch.float32 or tuple(feature.shape) != (batch, length, channels):
                raise ValueError('All feature layers must be aligned float32 patch tensors')
            if tuple(self.bank.patch_shapes[layer][1:]) != (length, channels):
                raise ValueError('Bank/query patch grid or feature dimension mismatch')
            query = F.normalize(feature, dim=-1)
            best = torch.full((batch, length), -torch.inf, dtype=torch.float32, device=self.device)
            for start_ref in range(0, len(unique_ids), rchunk):
                self._check()
                ids = unique_ids[start_ref:start_ref+rchunk]
                self.peak_reference_rows = max(self.peak_reference_rows, len(ids))
                raw = self.bank.read_patches(layer, ids.tolist())
                neighbors = F.normalize(raw.to(self.device), dim=-1)
                del raw
                for b in range(batch):
                    # Membership lookup is O(chunk*log(k)); no B*chunk*k mask.
                    offsets = torch.searchsorted(sorted_ids[b], ids)
                    matches = (offsets < k) & (sorted_ids[b][offsets.clamp(max=k-1)] == ids)
                    if not matches.any():
                        continue
                    selected = neighbors[matches.to(self.device)]
                    for start in range(0, length, qchunk):
                        self._check()
                        stop = min(start + qchunk, length)
                        if positional:
                            pos, valid = self._positions(start, stop, grid, radius, self.device)
                            local = selected[:, pos, :]
                            sims = torch.einsum('qc,kqpc->qkp', query[b, start:stop], local)
                            sims.masked_fill_(~valid[:, None, :], -torch.inf)
                            maximum = sims.flatten(1).max(dim=-1).values
                            del local, pos, valid
                        else:
                            sims = torch.matmul(query[b, start:stop], selected.reshape(-1, channels).t())
                            maximum = sims.max(dim=-1).values
                        self.peak_similarity_elements = max(self.peak_similarity_elements, sims.numel())
                        best[b, start:stop] = torch.maximum(best[b, start:stop], maximum)
                        del sims, maximum
                    del selected
                del neighbors
            if not torch.isfinite(best).all():
                raise ValueError('Non-finite patch scores; check checkpoint and bank')
            fused = fused + weights[li] * (1.0 - best)
            del query, best
        return fused, topk


def add_retrieval_arguments(parser):
    parser.add_argument('--reference-chunk-size', type=int, default=8)
    parser.add_argument('--query-patch-chunk-size', type=int, default=32)
    parser.add_argument('--cls-chunk-size', type=int, default=4096)
    parser.add_argument('--memory-workspace-mib', type=int, default=256)
    parser.add_argument('--max-score-mib', type=int, default=128)


def config_from_args(args):
    return RetrievalConfig(args.reference_chunk_size, args.query_patch_chunk_size,
                           args.cls_chunk_size, args.memory_workspace_mib, args.max_score_mib)
