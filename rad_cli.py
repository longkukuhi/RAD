"""Common CLI orchestration for version-2 RAD banks and resumable evaluation."""
from pathlib import Path
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from dataset import get_data_transforms, MVTecDataset, RealIADDataset
from dinov3.hub.backbones import load_dinov3_model
from rad_bank import BankReader, checkpoint_fingerprint, file_fingerprint, estimate_build_resources
from rad_evaluation import eval_patch_knn_multilayer_cluster, atomic_json, prepare_signature, load_cached_results
from rad_resources import add_resource_arguments, guard_from_args
from rad_retrieval import ChunkedRetriever, add_retrieval_arguments, config_from_args


def add_evaluation_arguments(parser):
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--metric-chunk-size', type=int, default=262144)
    parser.add_argument('--resume-eval', action='store_true', help='explicitly resume matching committed predictions')
    parser.add_argument('--cleanup-metric-spool', action='store_true', help='remove temporary maps only after successful metrics.json publication')
    add_retrieval_arguments(parser)
    add_resource_arguments(parser)


def dataset_signature(dataset, guard):
    digest = hashlib.sha256()
    max_pixels = 0
    for i in range(len(dataset)):
        guard.check()
        image = file_fingerprint(dataset.img_paths[i])
        with Image.open(image['path']) as handle:
            max_pixels = max(max_pixels, handle.width * handle.height)
        mask = None
        if int(dataset.labels[i]):
            mask = file_fingerprint(dataset.gt_paths[i])
            with Image.open(mask['path']) as handle:
                max_pixels = max(max_pixels, handle.width * handle.height)
        digest.update(json.dumps({'image': image, 'mask': mask, 'label': int(dataset.labels[i])},
                                 sort_keys=True).encode())
    return {'sha256': digest.hexdigest(), 'count': len(dataset), 'max_source_pixels': max_pixels}


def _budget(args, bank, device, max_source_pixels=0, metrics_only=False):
    if args.vis_max and not metrics_only:
        max_source_pixels = max(max_source_pixels, bank.provenance.get('max_source_image_pixels', 0))
    if metrics_only:
        estimate = {'ram_budget_bytes': 512 * 2**20, 'gpu_budget_bytes': 0,
                    'basis': 'CPU metric recovery; no encoder or GPU allocation', 'components': {}}
    else:
        estimate = estimate_build_resources(encoder_name=args.encoder_name,
            checkpoint_path=args.encoder_weight, batch_size=args.batch_size,
            image_size=512, crop_size=448, layers=bank.layers,
            num_workers=args.num_workers, device=device, max_source_pixels=max_source_pixels)
    # Sort temporary arrays (argsort, gathered records, groups, merge buffers).
    # Highly fragmented masks also create many regionprops Python objects,
    # dictionaries and cached region images, beyond the dense label/coord arrays.
    # This allowance is a conservative admission estimate, not an OS hard limit.
    aupro_image_bytes = args.resize_mask ** 2 * 1024
    metric_bytes = args.metric_chunk_size * 192 + aupro_image_bytes + 64 * 2**20
    retrieval_bytes = 0 if metrics_only else (args.memory_workspace_mib + args.max_score_mib) * 2**20
    mask_batches = 0 if metrics_only else (args.num_workers + 2) * args.batch_size * 448 ** 2 * 4
    # Resized maps/GT, Gaussian convolution workspace, full per-image sort
    # values+int64 indices and output copies coexist for the query batch. The
    # single-mask connected-component allowance above does not cover them.
    prediction_bytes = 0 if metrics_only else args.batch_size * args.resize_mask ** 2 * 128
    estimate['ram_budget_bytes'] += metric_bytes + retrieval_bytes + mask_batches + prediction_bytes
    if str(device).startswith('cuda'):
        estimate['gpu_budget_bytes'] += retrieval_bytes + prediction_bytes
    else:
        # Forward and retrieval buffers occur in the same CPU process; both
        # are already included once by the estimate plus retrieval_bytes above.
        pass
    estimate['components'].update(metric_workspace_bytes=metric_bytes,
        retrieval_workspace_bytes=retrieval_bytes, mask_prefetch_bytes=mask_batches,
        prediction_workspace_bytes=prediction_bytes, aupro_image_workspace_bytes=aupro_image_bytes,
        aupro_image_bytes_per_pixel=1024)
    estimate['basis'] += '; metric workspace is a conservative estimate, not an OS memory limit'
    return estimate


def _predictions_complete(directory):
    if (directory / 'metrics.json').is_file():
        return True
    state = directory / 'spool' / 'metric-state.json'
    if state.is_file():
        value = json.loads(state.read_text())
        return value['appended'] == value['num_samples']
    return False


def run_evaluation(args, realiad=False, print_fn=print):
    if args.batch_size <= 0 or args.num_workers < 0 or args.metric_chunk_size <= 0:
        raise ValueError('batch_size/metric-chunk-size must be positive and num_workers nonnegative')
    if args.resize_mask <= 0 or args.pos_radius < 0 or args.vis_max < 0:
        raise ValueError('Invalid evaluation dimensions or visualization count')
    if (not math.isfinite(args.max_ratio) or not 0 <= args.max_ratio <= 1 or
            (args.max_ratio and int(args.resize_mask ** 2 * args.max_ratio) < 1)):
        raise ValueError('max_ratio must select at least one pixel, or be 0 for max pooling')
    config = config_from_args(args)
    config.validate()
    run_dir = Path(args.save_dir) / args.save_name
    metrics_only = args.resume_eval and bool(args.item_list) and all(
        _predictions_complete(run_dir / 'metrics' / item) for item in args.item_list)
    device = 'cuda:0' if not metrics_only and torch.cuda.is_available() else 'cpu'
    print_fn(f'device: {device}')
    # The manifest and shape checks never load feature files into memory.
    bank = BankReader(args.bank_path, max_read_bytes=max(128, args.memory_workspace_mib) * 2**20)
    if args.layer_weights is not None and (len(args.layer_weights) != len(bank.layers)
            or not all(math.isfinite(w) for w in args.layer_weights)):
        raise ValueError('Provide one finite --layer_weights value per bank layer')
    estimate = _budget(args, bank, device, metrics_only=metrics_only)
    guard_args = SimpleNamespace(**vars(args))
    if metrics_only:
        guard_args.gpu_budget_gib = None  # A replayed GPU budget is irrelevant to CPU-only metric recovery.
    guard = guard_from_args(guard_args, estimate['ram_budget_bytes'], estimate['gpu_budget_bytes'],
                            device, str(run_dir / 'resources.jsonl'))
    guard.preflight()
    checkpoint_sha = checkpoint_fingerprint(args.encoder_weight)
    bank.validate_provenance({
        'encoder': {'name': args.encoder_name, 'checkpoint_sha256': checkpoint_sha},
        'preprocessing': {'image_size': 512, 'crop_size': 448,
                          'resize': 'square', 'interpolation': 'bilinear',
                          'order': ['Resize','ToTensor','CenterCrop','Normalize'],
                          'mean': [.485,.456,.406], 'std': [.229,.224,.225]},
        'feature_dtype': 'float32', 'seed': 1,
    })
    if not args.item_list or len(set(args.item_list)) != len(args.item_list):
        raise ValueError('item_list must contain unique categories')
    missing = set(args.item_list) - set(bank.provenance['categories'])
    if missing:
        raise ValueError(f'Categories are absent from this bank: {sorted(missing)}. Pass the correct --item_list.')
    if not 1 <= args.k_image <= bank.n_samples:
        raise ValueError(f'k_image must be between 1 and {bank.n_samples}')
    # Validate ranking capacity before loading the model or starting inference.
    score_bytes = args.batch_size * bank.n_samples * 4
    index_bytes = args.batch_size * args.k_image * (8 * 5 + 12)
    if not metrics_only and (score_bytes > args.max_score_mib * 2**20 or
            score_bytes * 9 + index_bytes > (args.max_score_mib + args.memory_workspace_mib) * 2**20):
        raise MemoryError('Scalar image-ranking buffers exceed the configured bound; adjust the measured resource budget and buffer caps before running')
    if not metrics_only:
        shape = bank.patch_shapes[bank.layers[0]]
        ChunkedRetriever(bank, device, config)._chunks(shape[1], shape[2],
            args.use_positional_bank, args.pos_radius, args.batch_size, args.batch_size * args.k_image * 8 * 5)
    data_transform, gt_transform = get_data_transforms(512,448)
    protocol = {
        'bank_manifest_sha256': checkpoint_fingerprint(Path(args.bank_path) / 'manifest.json'),
        'checkpoint_sha256': checkpoint_sha, 'encoder_name': args.encoder_name,
        'layers': bank.layers, 'image_size': 512, 'crop_size': 448, 'seed': 1,
        'k_image': args.k_image, 'resize_mask': args.resize_mask, 'max_ratio': args.max_ratio,
        'use_positional_bank': args.use_positional_bank, 'pos_radius': args.pos_radius,
        'layer_weights': args.layer_weights, 'dataset_kind': 'Real-IAD' if realiad else 'MVTec-format',
        'implementation': {name: checkpoint_fingerprint(Path(__file__).parent / name)
                           for name in ('rad_retrieval.py','rad_metrics.py','rad_evaluation.py','dataset.py')},
    }
    # Reject conflicting existing runs across every category before any model
    # allocation; per-category image fingerprints are checked immediately before
    # that category executes, without retaining all datasets in host memory.
    for item in args.item_list:
        existing_dir = run_dir / 'metrics' / item
        existing_inputs = existing_dir / 'evaluation-inputs.json'
        if existing_inputs.exists():
            if not args.resume_eval:
                raise FileExistsError('Existing evaluation categories require --resume-eval or a new save_name')
            saved = json.loads(existing_inputs.read_text())
            if any(saved.get(key) != value for key, value in {**protocol, 'category': item}.items()):
                raise ValueError('Existing evaluation protocol differs; use a new save_name')
        elif existing_dir.exists() and any(existing_dir.iterdir()):
            raise FileExistsError('Existing evaluation category has no provenance; use a new save_name')
    encoder = None
    all_results = []
    for item in args.item_list:
        guard.check()
        if realiad:
            test_data = RealIADDataset(root=args.data_path, category=item,
                transform=data_transform, gt_transform=gt_transform, phase='test')
        else:
            test_data = MVTecDataset(root=str(Path(args.data_path) / item),
                transform=data_transform, gt_transform=gt_transform, phase='test')
        if len(test_data) == 0:
            raise ValueError(f'Empty test category: {item}')
        signature = {**protocol, 'category': item, 'dataset': dataset_signature(test_data, guard)}
        metric_dir = run_dir / 'metrics' / item
        prepare_signature(metric_dir, signature, resume=args.resume_eval, create=False)
        estimate = _budget(args, bank, device, signature['dataset']['max_source_pixels'], metrics_only)
        print_fn('Resource estimate (not measured): ' + json.dumps(estimate, sort_keys=True))
        # Updating category/decode bounds does not reset the paging baseline.
        if args.ram_budget_gib is None:
            guard.ram_budget_bytes = estimate['ram_budget_bytes']
        if args.gpu_budget_gib is None:
            guard.gpu_budget_bytes = estimate['gpu_budget_bytes']
        guard.check(force=True)
        atomic_json(run_dir / f'{item}-execution.json', {
            'resource_estimate': estimate, 'retrieval': vars(config),
            'batch_size': args.batch_size, 'num_workers': args.num_workers,
            'metric_chunk_size': args.metric_chunk_size, 'resume': args.resume_eval,
            'metrics_only': metrics_only, 'gpu_budget_ignored_for_cpu_metrics': args.gpu_budget_gib if metrics_only else None,
        })
        # A completed matching category can be reused without allocating encoder.
        if args.resume_eval and (metric_dir / 'metrics.json').exists():
            prepare_signature(metric_dir, signature, resume=True)
            results = load_cached_results(metric_dir, cleanup_metric_spool=args.cleanup_metric_spool,
                                          guard=guard)
        else:
            if encoder is None and not _predictions_complete(metric_dir):
                guard.check(force=True)
                encoder = load_dinov3_model(args.encoder_name, pretrained_weight_path=args.encoder_weight)
                encoder = encoder.to(device).eval()
                for parameter in encoder.parameters():
                    parameter.requires_grad = False
                guard.check(force=True)
            kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                          pin_memory=False, drop_last=False)
            if args.num_workers:
                kwargs['prefetch_factor'] = 1
            loader = DataLoader(test_data, **kwargs)
            results = eval_patch_knn_multilayer_cluster(encoder, bank, loader, device,
                metric_dir=metric_dir, signature=signature, k_image=args.k_image,
                resize_mask=args.resize_mask, max_ratio=args.max_ratio,
                vis_dir=run_dir / 'test_vis_multilayer_cluster' / item, max_vis=args.vis_max,
                retrieve_transform=data_transform, data_root=args.data_path,
                use_positional_bank=args.use_positional_bank, pos_radius=args.pos_radius,
                layer_weights=args.layer_weights, retrieval_config=config, guard=guard,
                metric_chunk_size=args.metric_chunk_size, resume=args.resume_eval,
                cleanup_metric_spool=args.cleanup_metric_spool)
        all_results.append(results)
        print_fn(f'{item}: ' + ', '.join(f'{name}:{value:.4f}' for name,value in zip(
            ('I-AUROC','I-AP','I-F1','P-AUROC','P-AP','P-F1','P-AUPRO'), results)))
    mean = np.mean(all_results, axis=0).tolist()
    atomic_json(run_dir / 'summary.json', {'categories': args.item_list, 'results': all_results, 'mean': mean,
                                         'protocol': protocol})
    print_fn('Mean: ' + ', '.join(f'{name}:{value:.4f}' for name,value in zip(
        ('I-AUROC','I-AP','I-F1','P-AUROC','P-AP','P-F1','P-AUPRO'), mean)))
    return mean
