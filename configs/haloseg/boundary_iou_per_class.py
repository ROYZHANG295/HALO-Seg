"""
Per-Class Boundary IoU Calculator with Baseline vs HALO Comparison
"""

import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# =====================================================================
# === Configuration ===
# =====================================================================
# STEP 1: Generate predictions, e.g.:
# python tools/test.py experiments/halo-ddrnet_23-slim_in1k-pre-baseline_1xb12-120k_cityscapes-1024x1024-3090-77.85/halo-ddrnet_23-slim_in1k-pre-baseline_1xb12-120k_cityscapes-1024x1024.py experiments/halo-ddrnet_23-slim_in1k-pre-baseline_1xb12-120k_cityscapes-1024x1024-3090-77.85/best_mIoU_iter_114000.pth --out work_dirs/ddrnet-23-slim-baseline/predictions
# STEP2: python configs/halo_a4000/boundary_iou_per_class.py 

# BASELINE_DIR = "work_dirs/pidnet-s_baseline/predictions"
# HALO_DIR     = "work_dirs/pidnet-s_halo/predictions"
BASELINE_DIR = "work_dirs/ddrnet-23-slim-baseline/predictions"
HALO_DIR     = "work_dirs/ddrnet-23-slim-halo-best/predictions"
GT_DIR   = "data/cityscapes/gtFine/val"

NUM_CLASSES = 19
IGNORE_INDEX = 255
DILATION_RATIO = 0.02

# Official Cityscapes class names (19 classes)
CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
]

# For CamVid, use the following (11 classes):
# NUM_CLASSES = 11
# CLASS_NAMES = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk',
#                'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']


# =====================================================================
# === Core Functions ===
# =====================================================================

def mask_to_boundary(mask, dilation_ratio=0.02):
    """Convert a segmentation mask to its boundary region."""
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))

    new_mask = cv2.copyMakeBorder(
        mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1:h+1, 1:w+1]

    return mask - mask_erode


def boundary_iou_per_image(gt, pred, num_classes, ignore_index=255, dilation_ratio=0.02):
    """
    Compute per-class Boundary IoU for one image.
    Returns (intersection_array, union_array) for accumulation.
    """
    intersections = np.zeros(num_classes, dtype=np.float64)
    unions        = np.zeros(num_classes, dtype=np.float64)

    valid_mask = (gt != ignore_index)

    for cls in range(num_classes):
        gt_mask   = ((gt   == cls) & valid_mask).astype(np.uint8)
        pred_mask = ((pred == cls) & valid_mask).astype(np.uint8)

        if gt_mask.sum() == 0 and pred_mask.sum() == 0:
            continue

        gt_boundary   = mask_to_boundary(gt_mask,   dilation_ratio)
        pred_boundary = mask_to_boundary(pred_mask, dilation_ratio)

        intersections[cls] = (gt_boundary & pred_boundary).sum()
        unions[cls]        = (gt_boundary | pred_boundary).sum()

    return intersections, unions


# =====================================================================
# === Utility Functions ===
# =====================================================================

def find_gt_path(pred_filename, gt_root):
    """Find the Cityscapes GT path recursively under subdirectories."""
    base = pred_filename.replace('_leftImg8bit.png', '')
    for root, _, files in os.walk(gt_root):
        for f in files:
            if f.startswith(base) and 'labelTrainIds' in f:
                return os.path.join(root, f)
    return None


def evaluate_model(pred_dir, gt_dir, model_name=""):
    """Evaluate a single model and return per-class Boundary IoU."""
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    print(f"\n[{model_name}] Found {len(pred_files)} predictions")

    total_inter = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_union = np.zeros(NUM_CLASSES, dtype=np.float64)

    for pred_name in tqdm(pred_files, desc=f"Evaluating {model_name}"):
        pred_path = os.path.join(pred_dir, pred_name)

        # For Cityscapes, use find_gt_path; for CamVid, use the same filename.
        gt_path = find_gt_path(pred_name, gt_dir)
        # gt_path = os.path.join(gt_dir, pred_name)  # Use this line for CamVid.

        if gt_path is None or not os.path.exists(gt_path):
            continue

        pred = np.array(Image.open(pred_path))
        gt   = np.array(Image.open(gt_path))

        inter, union = boundary_iou_per_image(
            gt, pred, NUM_CLASSES, IGNORE_INDEX, DILATION_RATIO
        )
        total_inter += inter
        total_union += union

    # Compute IoU after accumulation for more stable results.
    per_class_biou = np.where(
        total_union > 0,
        total_inter / total_union,
        np.nan
    )
    return per_class_biou


# =====================================================================
# === Main Flow: Baseline vs HALO ===
# =====================================================================

def main():
    print("=" * 70)
    print("Per-Class Boundary IoU Comparison")
    print(f"Dilation ratio: {DILATION_RATIO}")
    print("=" * 70)

    biou_baseline = evaluate_model(BASELINE_DIR, GT_DIR, "Baseline")
    biou_halo     = evaluate_model(HALO_DIR,     GT_DIR, "HALO")

    delta = biou_halo - biou_baseline

    # Print comparison table.
    print("\n" + "=" * 70)
    print(f"{'Class':<18} {'Baseline':>10} {'HALO':>10} {'Δ':>10}")
    print("-" * 70)
    for cls in range(NUM_CLASSES):
        b = biou_baseline[cls] * 100
        h = biou_halo[cls]     * 100
        d = delta[cls]         * 100
        marker = "  ✅" if d > 0.5 else ("  ❌" if d < -0.5 else "")
        print(f"{CLASS_NAMES[cls]:<18} {b:>9.2f}% {h:>9.2f}% {d:>+9.2f}%{marker}")

    print("-" * 70)
    mean_b = np.nanmean(biou_baseline) * 100
    mean_h = np.nanmean(biou_halo)     * 100
    mean_d = mean_h - mean_b
    print(f"{'Mean Boundary IoU':<18} {mean_b:>9.2f}% {mean_h:>9.2f}% {mean_d:>+9.2f}%")
    print("=" * 70)

    # Sort classes by largest improvements.
    print("\n📈 Top 5 classes with LARGEST improvement:")
    valid_idx = ~np.isnan(delta)
    sorted_idx = np.argsort(-delta * valid_idx)
    for i in sorted_idx[:5]:
        if not np.isnan(delta[i]):
            print(f"  {CLASS_NAMES[i]:<18} {delta[i]*100:+.2f}%")

    print("\n📉 Top 5 classes with LARGEST drop:")
    sorted_idx = np.argsort(delta * valid_idx)
    for i in sorted_idx[:5]:
        if not np.isnan(delta[i]):
            print(f"  {CLASS_NAMES[i]:<18} {delta[i]*100:+.2f}%")


if __name__ == '__main__':
    main()
