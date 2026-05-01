"""
inference.py - with Test Time Augmentation (TTA: hflip + vflip)

Usage:
    python inference.py \
        --test_dir ./data/test_release \
        --id_json ./data/test_image_name_to_ids.json \
        --checkpoint ./checkpoints/last_model.pth \
        --output test-results.json \
        --score_thresh 0.5 \
        --nms_thresh 0.5 \
        --use_tta
"""

import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from pycocotools import mask as mask_utils
from torchvision.ops import nms as box_nms
import torchvision.transforms.functional as TF

from model import build_model
from dataset import TestDataset, get_val_transform


def encode_rle(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_bbox(binary_mask):
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def predict_single(model, image_tensor, device):
    """Run inference on one image tensor (C,H,W)."""
    with autocast(enabled=device.type == "cuda"):
        with torch.no_grad():
            out = model([image_tensor.to(device)])[0]
    return {k: v.cpu() for k, v in out.items()}


def merge_tta_outputs(outputs_list, h, w, score_thresh, nms_thresh):
    """
    Merge predictions from original + flipped images.
    outputs_list: list of dicts, each with boxes/scores/labels/masks
    flip_types:   list of 'none'/'hflip'/'vflip' matching outputs_list
    """
    all_boxes, all_scores, all_labels, all_masks = [], [], [], []
    flip_types = ["none", "hflip", "vflip"]

    for output, flip in zip(outputs_list, flip_types):
        scores = output["scores"].numpy()
        keep = scores >= score_thresh
        if not keep.any():
            continue

        boxes = output["boxes"][keep].numpy()
        scores = scores[keep]
        labels = output["labels"][keep].numpy()
        masks = output["masks"][keep].numpy()  # (N,1,H,W)

        # Un-flip boxes and masks
        if flip == "hflip":
            # box: x1,y1,x2,y2 → mirror horizontally
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            masks = masks[:, :, :, ::-1].copy()
        elif flip == "vflip":
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            masks = masks[:, :, ::-1, :].copy()

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_masks.append(masks)

    if len(all_boxes) == 0:
        return [], [], [], []

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # NMS to remove duplicates from TTA
    kept = box_nms(
        torch.from_numpy(all_boxes),
        torch.from_numpy(all_scores),
        iou_threshold=nms_thresh,
    ).numpy()

    return all_boxes[kept], all_scores[kept], all_labels[kept], all_masks[kept]


@torch.no_grad()
def run_inference(
    model,
    test_loader,
    image_name_to_id,
    device,
    score_thresh=0.5,
    nms_thresh=0.5,
    use_tta=True,
):
    model.eval()
    results = []

    for images, filenames in test_loader:
        img = images[0]  # batch_size=1, (C,H,W)
        fname = filenames[0]

        if fname not in image_name_to_id:
            print(f"  [WARNING] {fname} not in id mapping, skip.")
            continue

        image_id = image_name_to_id[fname]
        h, w = img.shape[1], img.shape[2]

        # ── TTA: original + hflip + vflip ──────────────────────────────
        if use_tta:
            imgs_aug = [
                img,
                TF.hflip(img),
                TF.vflip(img),
            ]
            outputs_list = [predict_single(model, im, device) for im in imgs_aug]
            boxes, scores, labels, masks = merge_tta_outputs(
                outputs_list, h, w, score_thresh, nms_thresh
            )
        else:
            output = predict_single(model, img, device)
            sc = output["scores"].numpy()
            keep = sc >= score_thresh
            if not keep.any():
                torch.cuda.empty_cache()
                continue
            boxes = output["boxes"][keep].numpy()
            scores = sc[keep]
            labels = output["labels"][keep].numpy()
            masks = output["masks"][keep].numpy()
            kept = box_nms(
                torch.from_numpy(boxes), torch.from_numpy(scores), nms_thresh
            ).numpy()
            boxes, scores, labels, masks = (
                boxes[kept],
                scores[kept],
                labels[kept],
                masks[kept],
            )

        for i in range(len(scores)):
            bmask = masks[i, 0] >= 0.5
            if bmask.sum() == 0:
                continue
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "segmentation": encode_rle(bmask),
                    "bbox": mask_to_bbox(bmask),
                    "score": float(scores[i]),
                }
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Total predictions: {len(results)}")
    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.id_json) as f:
        id_data = json.load(f)
    image_name_to_id = {item["file_name"]: item["id"] for item in id_data}

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone = ckpt.get("backbone", "resnet101")
    model = build_model(num_classes=5, backbone_name=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch','?')} backbone={backbone}")
    print(f"TTA: {'ON' if args.use_tta else 'OFF'}")

    test_dataset = TestDataset(args.test_dir, transform=get_val_transform())
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: tuple(zip(*b)),
    )

    results = run_inference(
        model,
        test_loader,
        image_name_to_id,
        device,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        use_tta=args.use_tta,
    )

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"Submission saved → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--id_json", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="test-results.json")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.5)
    parser.add_argument("--use_tta", action="store_true", default=True)
    parser.add_argument("--no_tta", dest="use_tta", action="store_false")
    args = parser.parse_args()
    main(args)
