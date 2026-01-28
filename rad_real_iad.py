import os
import random
import numpy as np
import warnings
import logging
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import get_data_transforms, RealIADDataset
from dinov3.hub.backbones import load_dinov3_model

from utils import compute_pro, f1_score_max, get_gaussian_kernel

from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def setup_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def tensor_to_np(img_tensor):
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.detach().cpu().unsqueeze(0)   # [1,3,H,W]
    img = img * IMAGENET_STD + IMAGENET_MEAN       # de-normalize
    img = img.clamp(0, 1)[0]
    img = img.permute(1, 2, 0).numpy()
    return img


@torch.no_grad()
def eval_patch_knn_multilayer_cluster(
    encoder,
    layer_indices,           # list[int], same order as in bank
    cls_banks,               # list[Tensor], each [N, C]
    patch_banks,             # list[Tensor], each [N, L, C]
    test_dataloader,
    device,
    k_image=5,
    resize_mask=256,
    max_ratio=0.0,
    vis_dir=None,
    max_vis=0,
    retrieve_transform=None,
    train_img_paths=None,
    use_positional_bank=False,
    pos_radius=1,
    layer_weights=None,      # list[float] or None
):
    """
    Multi-layer patch-KNN on Real-IAD.

    cls_banks: list[Tensor], each [N, C]
    patch_banks: list[Tensor], each [N, L, C]
    layer_weights: list[float], length = num_layers, if None then uniform weights
    """

    encoder.eval()
    device = torch.device(device)

    num_layers = len(layer_indices)
    assert len(cls_banks) == num_layers
    assert len(patch_banks) == num_layers

    # move banks to device
    cls_banks_dev = [cb.to(device) for cb in cls_banks]
    patch_banks_dev = [pb.to(device) for pb in patch_banks]

    # image-level nearest neighbors use the highest layer CLS
    cls_bank_global = cls_banks_dev[-1]            # [N, C]
    cls_bank_global_norm = F.normalize(cls_bank_global, dim=-1)

    # layer weights (default: uniform)
    if layer_weights is None or len(layer_weights) != num_layers:
        layer_weights = [1.0 / num_layers] * num_layers
    else:
        s = sum(layer_weights)
        if s <= 0:
            layer_weights = [1.0 / num_layers] * num_layers
        else:
            layer_weights = [w / s for w in layer_weights]

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=1.0).to(device)

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []

    remain_vis = max_vis
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    for img, gt, label, img_path in test_dataloader:
        img = img.to(device)
        gt = gt.to(device)

        # 1) extract multi-layer features
        inter = encoder.get_intermediate_layers(
            img,
            n=layer_indices,
            return_class_token=True,
            reshape=False,
            norm=True,
        )

        patch_list = []
        cls_list = []
        for (patch_tok, cls_tok) in inter:
            patch_list.append(patch_tok)   # [B, L, C]
            cls_list.append(cls_tok)       # [B, C]

        # image-level similarity with highest layer CLS
        cls_x_global = cls_list[-1]                        # [B, C]
        cls_x_global_norm = F.normalize(cls_x_global, dim=-1)
        sim_img = torch.matmul(cls_x_global_norm, cls_bank_global_norm.t())  # [B, N]
        _, topk_idx = torch.topk(sim_img, k_image, dim=-1)                   # [B, k_image]

        B, L, C = patch_list[0].shape
        h = w = int(L ** 0.5)

        # 2) multi-layer patch-KNN
        patch_scores_batch = []

        for b in range(B):
            neigh_indices = topk_idx[b]      # [k_image]

            scores_per_layer = []

            for li in range(num_layers):
                patches_x_l = patch_list[li]            # [B, L, C]
                bank_l = patch_banks_dev[li]           # [N, Lbank, C]

                q_feat = patches_x_l[b]                # [L, C]
                neigh_feat = bank_l[neigh_indices]     # [k_image, Lbank, C]

                q_feat = F.normalize(q_feat, dim=-1)           # [L, C]
                neigh_feat = F.normalize(neigh_feat, dim=-1)   # [k_image, Lbank, C]

                if use_positional_bank:
                    # position-aware KNN: assumes same patch grid size between
                    # bank and test features
                    patch_scores_l = []
                    for j in range(L):
                        r = j // w
                        c = j % w

                        r_min = max(0, r - pos_radius)
                        r_max = min(h - 1, r + pos_radius)
                        c_min = max(0, c - pos_radius)
                        c_max = min(w - 1, c + pos_radius)

                        idx_list = []
                        for rr in range(r_min, r_max + 1):
                            base = rr * w
                            for cc in range(c_min, c_max + 1):
                                idx_list.append(base + cc)

                        idx = torch.tensor(idx_list, device=device, dtype=torch.long)
                        neigh_local = neigh_feat[:, idx, :]          # [k_image, Kpos, C]
                        neigh_local = neigh_local.reshape(-1, C)     # [k_image*Kpos, C]

                        qj = q_feat[j:j + 1]                         # [1, C]
                        if neigh_local.numel() == 0:
                            patch_scores_l.append(torch.tensor(1.0, device=device))
                            continue

                        sim = torch.matmul(qj, neigh_local.t())      # [1, k_image*Kpos]
                        nn_sim, _ = sim.max(dim=-1)                  # [1]
                        patch_scores_l.append(1.0 - nn_sim.squeeze(0))

                    patch_score_l = torch.stack(patch_scores_l, dim=0)   # [L]
                else:
                    # global patch-KNN, no positional restriction
                    bank_local = neigh_feat.reshape(-1, C)              # [k_image*Lbank, C]
                    sim = torch.matmul(q_feat, bank_local.t())          # [L, k_image*Lbank]
                    nn_sim, _ = sim.max(dim=-1)                         # [L]
                    patch_score_l = 1.0 - nn_sim                        # [L]

                scores_per_layer.append(patch_score_l)

            # multi-layer linear fusion
            fused = torch.zeros_like(scores_per_layer[0])
            for li in range(num_layers):
                fused = fused + layer_weights[li] * scores_per_layer[li]

            patch_scores_batch.append(fused)

        patch_scores_batch = torch.stack(patch_scores_batch, dim=0)    # [B, L]

        # 3) reshape to patch map and upsample to pixel-level map
        patch_maps = patch_scores_batch.view(B, 1, h, w)  # [B,1,h,w]

        anomaly_map = F.interpolate(
            patch_maps,
            size=resize_mask,
            mode='bilinear',
            align_corners=False
        )  # [B,1,H',W']

        # 4) align GT and apply Gaussian smoothing
        gt = F.interpolate(gt, size=resize_mask, mode='nearest')
        anomaly_map = gaussian_kernel(anomaly_map)

        gt = gt.bool()
        if gt.shape[1] > 1:
            gt = torch.max(gt, dim=1, keepdim=True)[0]

        # 5) visualization
        if vis_dir is not None and remain_vis > 0:
            for b in range(B):
                if remain_vis <= 0:
                    break

                inp_np = tensor_to_np(img[b])
                amap_np = anomaly_map[b, 0].detach().cpu().numpy()
                gt_np = gt[b, 0].detach().cpu().numpy()

                # visualize nearest neighbor from highest layer CLS (optional)
                nn_idx = topk_idx[b, 0].item()
                nn_img_np = None
                if train_img_paths is not None:
                    nn_path = train_img_paths[nn_idx]
                    nn_pil = Image.open(nn_path).convert('RGB')
                    if retrieve_transform is not None:
                        nn_tensor = retrieve_transform(nn_pil)
                        nn_img_np = tensor_to_np(nn_tensor)
                    else:
                        nn_img_np = np.array(nn_pil)

                n_cols = 4 if nn_img_np is None else 5
                fig, axes = plt.subplots(1, n_cols, figsize=(15, 3))

                axes[0].imshow(inp_np)
                axes[0].set_title("Input")
                axes[0].axis('off')

                col = 1
                if nn_img_np is not None:
                    axes[1].imshow(nn_img_np)
                    axes[1].set_title(f"Nearest (idx={nn_idx})")
                    axes[1].axis('off')
                    col = 2

                im = axes[col].imshow(amap_np, cmap='jet')
                axes[col].set_title("Anomaly Map")
                axes[col].axis('off')
                fig.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

                axes[col + 1].imshow(gt_np, cmap='gray')
                axes[col + 1].set_title("GT mask")
                axes[col + 1].axis('off')

                plt.tight_layout()
                img_name = os.path.basename(img_path[b])
                save_path = os.path.join(vis_dir, f"vis_{img_name}")
                plt.savefig(save_path, dpi=150)
                plt.close()

                remain_vis -= 1

        # 6) collect metrics
        gt_list_px.append(gt)              # [B,1,H,W]
        pr_list_px.append(anomaly_map)     # [B,1,H,W]
        gt_list_sp.append(label)           # [B]

        # image-level score: max or top-ratio mean
        if max_ratio == 0:
            sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]       # [B]
        else:
            amap_flat = anomaly_map.flatten(1)
            k = int(amap_flat.shape[1] * max_ratio)
            sp_score = torch.sort(amap_flat, dim=1, descending=True)[0][:, :k].mean(dim=1)
        pr_list_sp.append(sp_score)

    # 7) compute AUROC / AUPRO etc.
    gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
    pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
    gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
    pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

    aupro_px = compute_pro(gt_list_px, pr_list_px)

    gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

    auroc_px = roc_auc_score(gt_list_px, pr_list_px)
    auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
    ap_px = average_precision_score(gt_list_px, pr_list_px)
    ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

    f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
    f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def main():
    parser = argparse.ArgumentParser(
        description='Multi-layer Patch-KNN on Real-IAD with DINOv3'
    )
    parser.add_argument('--data_path', type=str, default='../Real-IAD')
    parser.add_argument(
        '--bank_path',
        type=str,
        default='./bank/realiad_dinov3_vitb16_multilayer36911_448_bank.pth'
    )
    parser.add_argument('--encoder_name', type=str, default='dinov3_vitb16')
    parser.add_argument(
        '--encoder_weight',
        type=str,
        default='../weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    )
    parser.add_argument('--save_dir', type=str, default='./saved_results_realiad')
    parser.add_argument('--save_name', type=str, default='realiad_multilayer_36911_448')
    parser.add_argument(
        '--k_image',
        type=int,
        default=150,
        help='top-K nearest neighbor images for local patch bank'
    )
    parser.add_argument('--resize_mask', type=int, default=256)
    parser.add_argument(
        '--max_ratio',
        type=float,
        default=0.01,
        help='top ratio for image-level score (0 means use max)'
    )
    parser.add_argument(
        '--vis_max',
        type=int,
        default=8,
        help='max number of test visualizations per class'
    )

    parser.add_argument(
        '--use_positional_bank',
        action='store_true',
        help='use position-aware local patch bank (with neighborhood)'
    )
    parser.add_argument(
        '--pos_radius',
        type=int,
        default=1,
        help='neighborhood radius (in patch grid) for position-aware bank (0 = only same position)'
    )

    parser.add_argument(
        '--layer_weights',
        nargs='+',
        type=float,
        default=None,
        help='weights for each layer (same length as bank layers), default uniform'
    )

    # Real-IAD category list
    parser.add_argument(
        '--item_list',
        nargs='+',
        type=str,
        default=[
            'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
            'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
            'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
            'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
            'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper'
        ],
        help='list of Real-IAD object categories to evaluate'
    )

    args = parser.parse_args()

    setup_seed(1)

    os.makedirs(os.path.join(args.save_dir, args.save_name), exist_ok=True)
    logger = get_logger(
        args.save_name,
        os.path.join(args.save_dir, args.save_name)
    )
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(f"device: {device}")
    print_fn(
        f"use_positional_bank = {args.use_positional_bank}, "
        f"pos_radius = {args.pos_radius}"
    )
    print_fn(f"item_list = {args.item_list}")

    # test-time transforms (can differ from bank-building transforms if desired)
    image_size = 512
    crop_size = 448
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    # Only need test datasets for Real-IAD
    test_data_list = []
    for i, item in enumerate(args.item_list):
        test_data = RealIADDataset(
            root=args.data_path,
            category=item,
            transform=data_transform,
            gt_transform=gt_transform,
            phase="test"
        )
        test_data_list.append(test_data)

    # encoder and bank
    encoder = load_dinov3_model(
        args.encoder_name,
        pretrained_weight_path=args.encoder_weight
    )
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    bank_ckpt = torch.load(args.bank_path, map_location='cpu')
    layer_indices = bank_ckpt["layers"]          # list[int], 0-based block indices
    cls_banks = bank_ckpt["cls_banks"]           # list[Tensor], each [N, C]
    patch_banks = bank_ckpt["patch_banks"]       # list[Tensor], each [N, Lbank, C]

    num_layers = len(layer_indices)
    print_fn(f"Loaded multi-layer bank with {num_layers} layers: {layer_indices}")
    for li in range(num_layers):
        print_fn(
            f"  Layer {layer_indices[li]}: cls_bank {cls_banks[li].shape}, "
            f"patch_bank {patch_banks[li].shape}"
        )

    # layer weights
    if args.layer_weights is None:
        layer_weights = None
        print_fn("layer_weights: uniform")
    else:
        assert len(args.layer_weights) == num_layers, \
            f"len(layer_weights) must == {num_layers}"
        layer_weights = args.layer_weights
        print_fn(f"layer_weights: {layer_weights}")

    # per-class evaluation
    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

    for item, test_data in zip(args.item_list, test_data_list):
        test_dataloader = DataLoader(
            test_data,
            batch_size=8,
            shuffle=False,
            num_workers=4
        )

        vis_dir = os.path.join(
            args.save_dir, args.save_name, "test_vis_multilayer_cluster", item
        )

        results = eval_patch_knn_multilayer_cluster(
            encoder=encoder,
            layer_indices=layer_indices,
            cls_banks=cls_banks,
            patch_banks=patch_banks,
            test_dataloader=test_dataloader,
            device=device,
            k_image=args.k_image,
            resize_mask=args.resize_mask,
            max_ratio=args.max_ratio,
            vis_dir=vis_dir,
            max_vis=args.vis_max,
            retrieve_transform=None,
            train_img_paths=None,
            use_positional_bank=args.use_positional_bank,
            pos_radius=args.pos_radius,
            layer_weights=layer_weights,
        )

        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        auroc_sp_list.append(auroc_sp)
        ap_sp_list.append(ap_sp)
        f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px)
        ap_px_list.append(ap_px)
        f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)

        print_fn(
            f'{item} (MultiLayer-RealIAD): '
            f'I-AUROC:{auroc_sp:.4f}, I-AP:{ap_sp:.4f}, I-F1:{f1_sp:.4f}, '
            f'P-AUROC:{auroc_px:.4f}, P-AP:{ap_px:.4f}, P-F1:{f1_px:.4f}, P-AUPRO:{aupro_px:.4f}'
        )

    print_fn(
        'Mean (MultiLayer-RealIAD): '
        'I-AUROC:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
        'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
            np.mean(auroc_px_list), np.mean(ap_px_list),
            np.mean(f1_px_list), np.mean(aupro_px_list))
    )


if __name__ == '__main__':
    main()
