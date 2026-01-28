import os
import random
import numpy as np
import warnings
import argparse

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import ImageFolder

from dataset import get_data_transforms
from dinov3.hub.backbones import load_dinov3_model

warnings.filterwarnings("ignore")


def setup_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
):
    """
    Build an in-memory multi-layer feature bank using DINOv3 intermediate features.

    encoder.get_intermediate_layers(..., n=layer_idx_list, return_class_token=True)
    returns a list whose elements are (patch_tokens, cls_token):
        patch_tokens: [B, L, C]
        cls_token:    [B, C]

    Returns:
      layers:      list[int], sorted layer indices
      cls_banks:   list[Tensor[N, C]], aligned with layers
      patch_banks: list[Tensor[N, L, C]], aligned with layers
    """
    encoder.eval()
    layer_idx_list = sorted(list(layer_idx_list))
    num_layers = len(layer_idx_list)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=device.startswith("cuda"),
    )

    # Collect features per layer, then concatenate
    cls_feats = [[] for _ in range(num_layers)]
    patch_feats = [[] for _ in range(num_layers)]

    for step, (img, label, idx) in enumerate(loader):
        img = img.to(device, non_blocking=True)

        inter = encoder.get_intermediate_layers(
            img,
            n=layer_idx_list,
            return_class_token=True,
            reshape=False,
            norm=True,
        )

        # inter follows the same order as layer_idx_list
        for li, (patch_tok, cls_tok) in enumerate(inter):
            patch_feats[li].append(patch_tok.cpu())  # [B, L, C]
            cls_feats[li].append(cls_tok.cpu())      # [B, C]

        if step % 50 == 0:
            print(f"[Build RAM] step={step}/{len(loader)}")

    cls_banks = []
    patch_banks = []
    for li in range(num_layers):
        cls_bank_l = torch.cat(cls_feats[li], dim=0)       # [N, C]
        patch_bank_l = torch.cat(patch_feats[li], dim=0)   # [N, L, C]
        cls_banks.append(cls_bank_l)
        patch_banks.append(patch_bank_l)

    return layer_idx_list, cls_banks, patch_banks


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
                        default='./bank/mvtec_dinov3_vitb16_multilayer36911_224_bank.pth',
                        help='output .pth containing multi-layer banks')

    args = parser.parse_args()

    setup_seed(1)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device:", device)
    print("layers (0-based):", args.layer_idx_list)

    # Data augmentation (consistent with training)
    data_transform, _ = get_data_transforms(args.image_size, args.crop_size)

    # Build multi-category training set (all normal)
    train_data_list = []
    for i, item in enumerate(args.item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        train_data = ImageFolder(root=train_path, transform=data_transform)

        # Make labels non-conflicting and set them all to i for this item
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        train_data_list.append(train_data)

    train_data = ConcatDataset(train_data_list)
    indexed_train_data = IndexedDataset(train_data)
    print("Total train images:", len(train_data))

    # Load and freeze DINOv3 encoder
    encoder = load_dinov3_model(
        args.encoder_name,
        pretrained_weight_path=args.encoder_weight
    )
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # Build multi-layer memory bank (RAM)
    layers, cls_banks, patch_banks = build_memory_bank_multilayer(
        encoder=encoder,
        dataset=indexed_train_data,
        device=device,
        batch_size=args.batch_size,
        layer_idx_list=args.layer_idx_list,
        num_workers=args.num_workers,
    )

    # Print shapes
    for li, layer in enumerate(layers):
        print(f"layer={layer} cls_bank={tuple(cls_banks[li].shape)} "
              f"patch_bank={tuple(patch_banks[li].shape)}")

    os.makedirs(os.path.dirname(args.bank_path), exist_ok=True)
    torch.save(
        {
            "layers": layers,          # list[int]
            "cls_banks": cls_banks,    # list[Tensor[N, C]]
            "patch_banks": patch_banks # list[Tensor[N, L, C]]
        },
        args.bank_path
    )
    print(f"Saved multi-layer bank to {args.bank_path}")


if __name__ == '__main__':
    main()
