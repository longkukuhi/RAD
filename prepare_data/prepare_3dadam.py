import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _mkdirs_if_not_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def _copy_images(src_dir: Path, dst_dir: Path):
    if not src_dir.exists():
        return

    _mkdirs_if_not_exists(dst_dir)
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

    for p in src_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            shutil.copy2(p, dst_dir / p.name)


def _copy_masks(src_dir: Path, dst_dir: Path):
    if not src_dir.exists():
        return

    _mkdirs_if_not_exists(dst_dir)
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

    for p in src_dir.iterdir():
        if not (p.is_file() and p.suffix.lower() in exts):
            continue

        dst_path = dst_dir / p.name
        shutil.copy2(p, dst_path)


def convert_category(src_root: Path, dst_root: Path, category: str):
    src_cat = src_root / category

    if not (src_cat / "train").is_dir():
        return

    dst_cat = dst_root / category

    # train/good
    train_good_src = src_cat / "train" / "good" / "rgb"
    train_good_dst = dst_cat / "train" / "good"
    _copy_images(train_good_src, train_good_dst)

    # validation/good -> test/good
    val_good_src = src_cat / "validation" / "good" / "rgb"
    test_good_dst = dst_cat / "test" / "good"
    _copy_images(val_good_src, test_good_dst)

    # test defects
    test_root = src_cat / "test"
    if not test_root.is_dir():
        return

    for defect_name in sorted(os.listdir(test_root)):
        defect_dir = test_root / defect_name
        if not defect_dir.is_dir():
            continue

        rgb_dir = defect_dir / "rgb"
        gt_dir = defect_dir / "ground_truth"

        test_defect_dst = dst_cat / "test" / defect_name
        gt_defect_dst = dst_cat / "ground_truth" / defect_name

        _copy_images(rgb_dir, test_defect_dst)
        _copy_masks(gt_dir, gt_defect_dst)


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D-ADAM_anomalib dataset to MVTec-style format"
    )
    parser.add_argument(
        "--data-folder",
        default="/path/to/3D-ADAM_anomalib",
        type=str,
        help="path to original 3D-ADAM_anomalib dataset",
    )
    parser.add_argument(
        "--save-folder",
        default="/path/to/mvtec-style-dataset",
        type=str,
        help="target path to save the MVTec-style dataset",
    )

    config = parser.parse_args()

    src_root = Path(config.data_folder)
    dst_root = Path(config.save_folder)

    _mkdirs_if_not_exists(dst_root)

    for name in sorted(os.listdir(src_root)):
        cat_path = src_root / name
        if not cat_path.is_dir():
            continue
        if not (cat_path / "train").is_dir():
            continue

        print(f"Converting category: {name}")
        convert_category(src_root, dst_root, name)

    print("Conversion finished.")


if __name__ == "__main__":
    main()
