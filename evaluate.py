"""
evaluate.py - Local COCO AP50 evaluation
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from model import build_model
from dataset import CellDataset, get_val_transform, CLASS_NAMES


def encode_rle(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def build_coco_gt(dataset, indices):
    images, annotations, ann_id = [], [], 1
    for img_id, idx in enumerate(indices, start=1):
        img, target = dataset[idx]
        h, w = img.shape[1], img.shape[2]
        images.append({"id": img_id, "height": h, "width": w})
        masks = target["masks"].numpy()
        labels = target["labels"].numpy()
        for j in range(len(masks)):
            bmask = masks[j]
            if bmask.sum() == 0:
                continue
            rle = encode_rle(bmask)
            ys, xs = np.where(bmask)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(labels[j]),
                    "segmentation": rle,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "area": int(bmask.sum()),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    categories = [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES, start=1)]
    return {"images": images, "annotations": annotations, "categories": categories}


@torch.no_grad()
def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Mode: {args.mode}")

    full_dataset = CellDataset(args.data_root, transforms=get_val_transform())
    n = len(full_dataset)

    if args.mode == "val":
        # 舊模式：切 15% 當 val（seed 要跟 train.py 一致）
        val_n = max(1, int(0.15 * n))
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
        eval_indices = idx[:val_n]
        print(f"Val split mode: {len(eval_indices)} images")
    else:
        # full 模式：用全部資料
        eval_indices = list(range(n))
        print(f"Full mode: all {n} images")
        print(
            "  ⚠️  注意：train on all → eval on all 會高估真實分數，僅供確認模型是否學到東西"
        )

    # Build COCO GT
    gt_dict = build_coco_gt(full_dataset, eval_indices)
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone = ckpt.get("backbone", "resnet101")
    model = build_model(num_classes=5, backbone_name=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model: {backbone}  epoch={ckpt.get('epoch','?')}")

    loader = DataLoader(
        Subset(full_dataset, eval_indices),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: tuple(zip(*b)),
    )

    results = []
    for img_id, (images, _) in enumerate(loader, start=1):
        images_gpu = [img.to(device) for img in images]
        with autocast(enabled=device.type == "cuda"):
            outputs = model(images_gpu)
        output = outputs[0]

        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        masks = output["masks"].cpu().numpy()

        keep = scores >= args.score_thresh
        if not keep.any():
            continue

        for i in np.where(keep)[0]:
            bmask = masks[i, 0] >= 0.5
            if bmask.sum() == 0:
                continue
            rle = encode_rle(bmask)
            ys, xs = np.where(bmask)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            results.append(
                {
                    "image_id": img_id,
                    "category_id": int(labels[i]),
                    "segmentation": rle,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(scores[i]),
                }
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(results) == 0:
        print("No predictions — lower --score_thresh")
        return

    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    ap50 = evaluator.stats[1]
    print(f"\n>>> AP50: {ap50:.4f} <<<")
    if args.mode == "full":
        print("  (這是 train set AP，實際 test AP 會較低)")
    return ap50


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "val"],
        help="full=用全部訓練資料評估, val=切15%%當val（舊模式）",
    )
    args = parser.parse_args()
    run_eval(args)
