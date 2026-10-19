import os
import random
import numpy as np
import warnings
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from dataset import get_data_transforms
from dinov3.hub.backbones import load_dinov3_model
from rad_bank import BankWriter, build_provenance, dataset_reference, estimate_build_resources
from rad_resources import add_resource_arguments, guard_from_args

warnings.filterwarnings("ignore")


def setup_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_normal_training_split(dataset, category, *, realiad=False):
    """Reject a mixed/incorrect train split; never silently drop references."""
    if realiad:
        if any(label != 0 for label in dataset.labels):
            raise ValueError(f'{category}: Real-IAD train contains anomalous labels; '
                             'verify the declared training split before building a normal reference bank. '
                             'No samples were filtered.')
    elif list(dataset.classes) != ['good']:
        raise ValueError(f'{category}: the normal training ImageFolder must contain only the good class; '
                         f'found {dataset.classes!r}. Verify the dataset preparation/split. '
                         'No samples were filtered.')


class IndexedDataset(Dataset):
    """
    Wrap a base Dataset so that __getitem__ returns (img, label, idx).
    Useful for index-based operations (e.g., excluding self in retrieval).
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label, idx


@torch.no_grad()
def build_memory_bank_multilayer(
    encoder,
    dataset,
    device,
    batch_size: int = 32,
    layer_idx_list=(3, 6, 9, 11),
    num_workers: int = 4,
    *,
    writer: BankWriter,
    reference_for_index,
    guard,
):
    """
    Stream every float32 feature row to a version-2 bank and return BankReader.
    A resumed build skips only already committed rows, preserving original order.
    """
    encoder.eval()
    layer_idx_list = sorted(list(layer_idx_list))
    if writer.layers != layer_idx_list or writer.n_samples != len(dataset):
        raise ValueError('Writer layers/count must match extraction dataset')
    if writer.written_samples == len(dataset):
        return writer.finalize()
    remaining = Subset(dataset, range(writer.written_samples, len(dataset)))

    loader = DataLoader(
        remaining,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        **({'prefetch_factor': 1} if num_workers else {}),
    )

    for step, (img, label, idx) in enumerate(loader):
        guard.check()
        expected = list(range(writer.written_samples, writer.written_samples + len(idx)))
        if idx.tolist() != expected:
            raise ValueError('DataLoader reference order differs from the committed bank')
        records = [reference_for_index(index) for index in expected]
        img = img.to(device)

        inter = encoder.get_intermediate_layers(
            img,
            n=layer_idx_list,
            return_class_token=True,
            reshape=False,
            norm=True,
        )

        if len(inter) != len(layer_idx_list):
            raise ValueError('Encoder did not return all requested layers')
        writer.append({layer: inter[li][1] for li, layer in enumerate(layer_idx_list)},
                      {layer: inter[li][0] for li, layer in enumerate(layer_idx_list)}, records)
        del inter, img, records
        guard.check()

        if step % 50 == 0:
            print(f"[Build disk bank] committed={writer.written_samples}/{writer.n_samples}")
    return writer.finalize()


def main():
    parser = argparse.ArgumentParser(
        description="Build MULTI-LAYER memory bank for MVTEC using DINOv3 encoder"
    )
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--item_list', nargs='+', default=[
        'carpet', 'grid', 'leather', 'tile', 'wood',
        'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
        'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
    ])
    parser.add_argument('--encoder_name', type=str, default='dinov3_vitb16')
    parser.add_argument('--encoder_weight', type=str,
                        default='../weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--layer_idx_list', nargs='+', type=int,
                        default=[3, 6, 9, 11],
                        help='0-based block indices for multi-layer bank, e.g. 3 6 9 11')

    parser.add_argument('--bank_path', type=str,
                        default='./bank/mvtec_dinov3_vitb16_multilayer36911_448_bank',
                        help='new version-2 bank directory (existing files are never overwritten)')
    parser.add_argument('--resume-bank', action='store_true', help='explicitly resume a verified incomplete bank')
    add_resource_arguments(parser)

    args = parser.parse_args()

    setup_seed(1)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device:", device)
    print("layers (0-based):", args.layer_idx_list)

    estimate = estimate_build_resources(encoder_name=args.encoder_name, checkpoint_path=args.encoder_weight,
        batch_size=args.batch_size, image_size=args.image_size, crop_size=args.crop_size,
        layers=args.layer_idx_list, num_workers=args.num_workers, device=device)
    guard = guard_from_args(args, ram_budget_bytes=estimate['ram_budget_bytes'],
                            gpu_budget_bytes=estimate['gpu_budget_bytes'], device=device,
                            log_path=str(Path(args.bank_path).with_suffix('.resources.jsonl')))
    guard.preflight()

    # Data augmentation (consistent with training)
    data_transform, _ = get_data_transforms(args.image_size, args.crop_size)

    # Build multi-category training set (all normal)
    train_data_list = []
    for i, item in enumerate(args.item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        train_data = ImageFolder(root=train_path, transform=data_transform)
        require_normal_training_split(train_data, item)

        # Make labels non-conflicting and set them all to i for this item
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        train_data_list.append(train_data)

    train_data = ConcatDataset(train_data_list)
    indexed_train_data = IndexedDataset(train_data)
    print("Total train images:", len(train_data))

    reference = lambda index: dataset_reference(indexed_train_data, index, args.item_list, args.data_path)
    provenance = build_provenance(encoder_name=args.encoder_name, checkpoint_path=args.encoder_weight,
                                  image_size=args.image_size, crop_size=args.crop_size, categories=args.item_list,
                                  reference_records=(reference(i) for i in range(len(train_data))),
                                  source_file=__file__, guard=guard)
    estimate = estimate_build_resources(encoder_name=args.encoder_name, checkpoint_path=args.encoder_weight,
        batch_size=args.batch_size, image_size=args.image_size, crop_size=args.crop_size,
        layers=args.layer_idx_list, num_workers=args.num_workers, device=device,
        max_source_pixels=provenance['max_source_image_pixels'])
    print('Resource estimate (CLI measured budgets override):', json.dumps(estimate, sort_keys=True))
    guard = guard_from_args(args, ram_budget_bytes=estimate['ram_budget_bytes'],
                            gpu_budget_bytes=estimate['gpu_budget_bytes'], device=device,
                            log_path=str(Path(args.bank_path).with_suffix('.resources.jsonl')))
    guard.preflight()
    with BankWriter(args.bank_path, n_samples=len(train_data), layers=sorted(args.layer_idx_list),
                    provenance=provenance, resume=args.resume_bank, guard=guard) as writer:
        if writer.written_samples == len(train_data):
            reader = writer.finalize()
        else:
            guard.check(force=True)
            encoder = load_dinov3_model(args.encoder_name, pretrained_weight_path=args.encoder_weight)
            encoder.to(device)
            for parameter in encoder.parameters():
                parameter.requires_grad = False
            encoder.eval()
            guard.check(force=True)
            reader = build_memory_bank_multilayer(encoder, indexed_train_data, device,
                batch_size=args.batch_size, layer_idx_list=args.layer_idx_list, num_workers=args.num_workers,
                writer=writer, reference_for_index=reference, guard=guard)
    for layer in reader.layers:
        print(f'layer={layer} cls_bank={reader.cls_shapes[layer]} patch_bank={reader.patch_shapes[layer]}')
    print(f"Saved multi-layer bank to {args.bank_path}")


if __name__ == '__main__':
    main()
