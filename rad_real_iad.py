import argparse
import logging
import os
import random

import numpy as np
import torch

from rad_cli import add_evaluation_arguments, run_evaluation
from rad_evaluation import eval_patch_knn_multilayer_cluster

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

def main():
    parser = argparse.ArgumentParser(
        description='Multi-layer Patch-KNN on Real-IAD with DINOv3'
    )
    parser.add_argument('--data_path', type=str, default='../Real-IAD')
    parser.add_argument(
        '--bank_path',
        type=str,
        default='./bank/realiad_dinov3_vitb16_multilayer36911_448_bank'
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

    add_evaluation_arguments(parser)
    args = parser.parse_args()
    setup_seed(1)
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    return run_evaluation(args, realiad=True, print_fn=logger.info)


if __name__ == "__main__":
    main()
