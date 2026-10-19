import os
import warnings
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from dataset import get_data_transforms, RealIADDataset
from dinov3.hub.backbones import load_dinov3_model
from build_bank_multilayer import IndexedDataset, build_memory_bank_multilayer, require_normal_training_split, setup_seed
from rad_bank import BankWriter, build_provenance, dataset_reference, estimate_build_resources
from rad_resources import add_resource_arguments, guard_from_args

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="Build MULTI-LAYER memory bank for Real-IAD using DINOv3 encoder"
    )
    parser.add_argument('--data_path', type=str, default='../Real-IAD')

    parser.add_argument('--item_list', nargs='+', default=[
        'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
        'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
        'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
        'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
        'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper'
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
                        default='./bank/realiad_dinov3_vitb16_multilayer36911_448_bank',
                        help='new version-2 bank directory (existing files are never overwritten)')
    parser.add_argument('--resume-bank', action='store_true', help='explicitly resume a verified incomplete bank')
    add_resource_arguments(parser)

    args = parser.parse_args()

    setup_seed(1)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device:", device)
    print("layers (0-based):", args.layer_idx_list)
    print("item_list:", args.item_list)

    estimate = estimate_build_resources(encoder_name=args.encoder_name, checkpoint_path=args.encoder_weight,
        batch_size=args.batch_size, image_size=args.image_size, crop_size=args.crop_size,
        layers=args.layer_idx_list, num_workers=args.num_workers, device=device)
    guard = guard_from_args(args, ram_budget_bytes=estimate['ram_budget_bytes'],
                            gpu_budget_bytes=estimate['gpu_budget_bytes'], device=device,
                            log_path=str(Path(args.bank_path).with_suffix('.resources.jsonl')))
    guard.preflight()

    # Data augmentation (consistent with training)
    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)

    # Build multi-category training set (Real-IAD, phase='train')
    train_data_list = []
    for i, item in enumerate(args.item_list):
        train_data = RealIADDataset(
            root=args.data_path,
            category=item,
            transform=data_transform,
            gt_transform=gt_transform,
            phase='train'
        )
        require_normal_training_split(train_data, item, realiad=True)
        # Give each category a unique label id i
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data_list.append(train_data)

    train_data = ConcatDataset(train_data_list)
    indexed_train_data = IndexedDataset(train_data)
    print("Total train images (Real-IAD):", len(train_data))

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
