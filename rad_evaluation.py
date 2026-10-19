"""Shared RAD evaluator: bounded retrieval, resumable maps, exact metrics."""
from pathlib import Path
import hashlib
import json
import math
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler, Subset

from rad_metrics import MetricAccumulator
from rad_retrieval import ChunkedRetriever


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_cached_results(directory, *, cleanup_metric_spool=False, guard=None):
    """Load published metrics and honor later, explicit spool cleanup.

    Only a successfully completed MetricAccumulator spool may be removed.
    Unrelated files and the published metrics/provenance remain intact, and
    requesting cleanup again after the owned spool was removed is harmless.
    """
    directory = Path(directory)
    values = json.loads((directory / 'metrics.json').read_text())['values']
    spool_path = directory / 'spool'
    if cleanup_metric_spool and (spool_path / 'metric-state.json').is_file():
        if guard is not None:
            guard.check()
        with MetricAccumulator.resume(spool_path, guard=guard) as metrics:
            metrics.cleanup()
    return values


def gaussian_kernel(device):
    # Identical kernel, float32 construction, padding and normalization to utils.
    size, sigma = 5, 1.0
    coord = torch.arange(size)
    x_grid = coord.repeat(size).view(size, size)
    xy_grid = torch.stack([x_grid, x_grid.t()], dim=-1).float()
    weights = (1. / (2. * math.pi * sigma ** 2)) * torch.exp(
        -torch.sum((xy_grid - (size - 1) / 2.) ** 2., dim=-1) / (2 * sigma ** 2))
    weights = weights / weights.sum()
    return weights.view(1, 1, size, size).to(device)


def _image_array(tensor):
    tensor = tensor.detach().cpu()
    mean = torch.tensor([.485, .456, .406]).view(3,1,1)
    std = torch.tensor([.229, .224, .225]).view(3,1,1)
    return (tensor * std + mean).clamp(0,1).permute(1,2,0).numpy()


def _visualize(image, amap, mask, path, neighbor_index, bank, data_root, transform, directory):
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from rad_bank import file_fingerprint
    reference = bank.read_reference(neighbor_index)
    relative = reference.get('relative_path')
    candidate = Path(data_root) / relative if data_root is not None and relative else Path(reference['path'])
    # Visualization paths come from the bank, not a freshly reordered ImageFolder.
    if file_fingerprint(candidate)['sha256'] != reference['sha256']:
        raise ValueError(f'Reference image changed since bank creation: {candidate}')
    with Image.open(candidate) as source:
        neighbor = _image_array(transform(source.convert('RGB')))
    fig, axes = plt.subplots(1,4,figsize=(13,3))
    try:
        axes[0].imshow(_image_array(image)); axes[0].set_title('Input')
        axes[1].imshow(neighbor); axes[1].set_title(f'Nearest (idx={neighbor_index})')
        plot = axes[2].imshow(amap, cmap='jet'); axes[2].set_title('Anomaly Map')
        fig.colorbar(plot, ax=axes[2], fraction=.046, pad=.04)
        axes[3].imshow(mask, cmap='gray'); axes[3].set_title('GT mask')
        for axis in axes:
            axis.axis('off')
        fig.tight_layout()
        # Different defect subfolders often contain identical basenames.
        suffix = hashlib.sha256(str(path).encode()).hexdigest()[:10]
        fig.savefig(Path(directory) / f'vis_{suffix}_{Path(path).name}', dpi=150)
    finally:
        plt.close(fig)


def prepare_signature(directory, signature, resume, create=True):
    directory = Path(directory)
    path = directory / 'evaluation-inputs.json'
    if path.exists():
        if not resume:
            raise FileExistsError(f'Existing evaluation: {directory}. Use a new save_name or --resume-eval.')
        if json.loads(path.read_text()) != signature:
            raise ValueError('Evaluation inputs/protocol changed; use a new run directory')
    else:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError('Existing evaluation files have no matching provenance; use a new directory')
        if create:
            atomic_json(path, signature)


@torch.no_grad()
def eval_patch_knn_multilayer_cluster(
    encoder, bank, test_dataloader, device, *, metric_dir, signature,
    k_image=150, resize_mask=256, max_ratio=.01, vis_dir=None, max_vis=0,
    retrieve_transform=None, data_root=None, use_positional_bank=False,
    pos_radius=1, layer_weights=None, retrieval_config=None, guard=None,
    metric_chunk_size=262144, resume=False, cleanup_metric_spool=False,
):
    if not isinstance(test_dataloader.sampler, SequentialSampler) or test_dataloader.drop_last:
        raise ValueError('Evaluation/resume requires a sequential DataLoader with drop_last=False')
    if resize_mask <= 0 or max_vis < 0 or not 0 <= max_ratio <= 1:
        raise ValueError('Invalid resize_mask, vis_max, or max_ratio')
    if max_ratio and int(resize_mask ** 2 * max_ratio) < 1:
        raise ValueError('max_ratio selects zero pixels; use 0 for max pooling or a larger ratio')
    if not 1 <= k_image <= bank.n_samples:
        raise ValueError('k_image exceeds the available reference images')
    directory = Path(metric_dir)
    prepare_signature(directory, signature, resume)
    result_path = directory / 'metrics.json'
    if resume and result_path.exists():
        return load_cached_results(directory, cleanup_metric_spool=cleanup_metric_spool, guard=guard)
    spool_path = directory / 'spool'
    state_path = spool_path / 'metric-state.json'
    if state_path.exists():
        if not resume:
            raise FileExistsError('Metric spool exists; use --resume-eval')
        metrics = MetricAccumulator.resume(spool_path, chunk_size=metric_chunk_size, guard=guard)
    else:
        metrics = MetricAccumulator(spool_path, len(test_dataloader.dataset), resize_mask, resize_mask,
                                    chunk_size=metric_chunk_size, guard=guard)
    if (metrics.num_samples, metrics.height, metrics.width) != (len(test_dataloader.dataset), resize_mask, resize_mask):
        metrics.close()
        raise ValueError('Metric spool shape differs from the dataset/protocol')
    initial_count = metrics.appended
    if initial_count:
        loader_kwargs = dict(batch_size=test_dataloader.batch_size, shuffle=False,
                             num_workers=test_dataloader.num_workers, pin_memory=False,
                             collate_fn=test_dataloader.collate_fn)
        if test_dataloader.num_workers:
            loader_kwargs['prefetch_factor'] = 1
        test_dataloader = DataLoader(Subset(test_dataloader.dataset,
                                            range(initial_count, len(test_dataloader.dataset))), **loader_kwargs)
    if metrics.appended < metrics.num_samples:
        if encoder is None:
            metrics.close()
            raise ValueError('An encoder is required for uncomputed test samples')
        encoder.eval()
    retriever = ChunkedRetriever(bank, device, retrieval_config, guard)
    kernel = gaussian_kernel(device) if metrics.appended < metrics.num_samples else None
    remaining_vis = max(0, max_vis - initial_count)
    if vis_dir and remaining_vis:
        Path(vis_dir).mkdir(parents=True, exist_ok=True)
    try:
        batches = () if metrics.appended == metrics.num_samples else test_dataloader
        for img, gt, label, img_path in batches:
            if guard:
                guard.check()
            img, gt = img.to(device), gt.to(device)
            features = encoder.get_intermediate_layers(img, n=bank.layers,
                return_class_token=True, reshape=False, norm=True)
            patch_features = [x[0] for x in features]
            cls_features = [x[1] for x in features]
            scores, topk = retriever.score_batch(patch_features, cls_features, k_image,
                use_positional_bank, pos_radius, layer_weights)
            b, length = scores.shape
            grid = math.isqrt(length)
            anomaly_map = F.interpolate(scores.view(b,1,grid,grid), size=resize_mask,
                                        mode='bilinear', align_corners=False)
            anomaly_map = F.conv2d(anomaly_map, kernel, padding=2)
            gt = F.interpolate(gt, size=resize_mask, mode='nearest').bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True).values
            flat = anomaly_map.flatten(1)
            if max_ratio == 0:
                sample_scores = flat.max(dim=1).values
            else:
                count = int(flat.shape[1] * max_ratio)
                sample_scores = flat.sort(dim=1, descending=True).values[:, :count].mean(dim=1)
            masks_cpu = gt[:,0].cpu().numpy()
            maps_cpu = anomaly_map[:,0].cpu().numpy()
            scores_cpu = sample_scores.cpu().numpy()
            if vis_dir and remaining_vis:
                for index in range(min(b, remaining_vis)):
                    if guard:
                        guard.check()
                    _visualize(img[index], maps_cpu[index], masks_cpu[index], img_path[index],
                               topk[index,0].item(), bank, data_root, retrieve_transform, vis_dir)
                    remaining_vis -= 1
            metrics.append(masks_cpu, maps_cpu, label.detach().cpu().numpy(), scores_cpu)
            del img, gt, features, patch_features, cls_features, scores, topk
            del anomaly_map, sample_scores, masks_cpu, maps_cpu, scores_cpu, flat
        results = metrics.compute()
        atomic_json(result_path, {'names': ['I-AUROC','I-AP','I-F1','P-AUROC','P-AP','P-F1','P-AUPRO'],
                                  'values': results, 'sample_count': metrics.num_samples,
                                  'retrieval_peak_reference_rows': retriever.peak_reference_rows,
                                  'retrieval_peak_similarity_elements': retriever.peak_similarity_elements})
        if cleanup_metric_spool:
            metrics.cleanup()
        return results
    except BaseException as error:
        atomic_json(directory / 'interrupted.json', {
            'status': 'interrupted', 'error_type': type(error).__name__, 'message': str(error),
            'committed_samples': metrics.appended, 'total_samples': metrics.num_samples,
            'resume': 'Re-run the same scientific configuration with --resume-eval after resolving the cause.',
        })
        raise
    finally:
        metrics.close()
