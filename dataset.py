"""
dataset.py - with stronger augmentation to combat overfitting
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_NAMES = ["class1", "class2", "class3", "class4"]
NUM_CLASSES = 4


def get_train_transform():
    return A.Compose(
        [
            # Geometry
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.5
            ),
            # Color
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # Noise / blur
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Sharpen(p=0.2),
            # Cutout / dropout
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                fill_value=0,
                p=0.3,
            ),
            # Normalize
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_transform():
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class CellDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.samples = []

        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            img_path = os.path.join(folder_path, "image.tif")
            if not os.path.exists(img_path):
                continue
            class_paths = {}
            for cls in CLASS_NAMES:
                p = os.path.join(folder_path, f"{cls}.tif")
                if os.path.exists(p):
                    class_paths[cls] = p
            self.samples.append((img_path, class_paths))

        print(f"[Dataset] Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_paths = self.samples[idx]

        image = tifffile.imread(img_path)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        image = image.astype(np.uint8)

        masks, labels = [], []
        for cls_idx, cls_name in enumerate(CLASS_NAMES, start=1):
            if cls_name not in class_paths:
                continue
            mask = tifffile.imread(class_paths[cls_name])
            for iid in np.unique(mask):
                if iid == 0:
                    continue
                binary_mask = (mask == iid).astype(np.uint8)
                if binary_mask.sum() < 10:
                    continue
                masks.append(binary_mask)
                labels.append(cls_idx)

        if len(masks) == 0:
            masks = [np.zeros(image.shape[:2], dtype=np.uint8)]
            labels = [1]

        masks_np = np.stack(masks, axis=0)

        if self.transforms:
            transformed = self.transforms(
                image=image, masks=[masks_np[i] for i in range(len(masks_np))]
            )
            image = transformed["image"]
            masks_tensor = torch.stack(
                [
                    torch.as_tensor(np.array(m), dtype=torch.uint8)
                    for m in transformed["masks"]
                ]
            )
        else:
            image = (
                torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            )
            masks_tensor = torch.as_tensor(masks_np, dtype=torch.uint8)

        boxes, keep = [], []
        for i in range(len(masks_tensor)):
            m = masks_tensor[i].numpy()
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            if x2 > x1 and y2 > y1:
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                keep.append(i)

        if len(keep) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, *masks_tensor.shape[1:]), dtype=torch.uint8)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor([labels[i] for i in keep], dtype=torch.int64)
            masks_t = masks_tensor[keep]

        return image, {"boxes": boxes_t, "labels": labels_t, "masks": masks_t}


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = sorted(
            [
                os.path.join(test_dir, f)
                for f in os.listdir(test_dir)
                if f.endswith(".tif")
            ]
        )
        print(f"[TestDataset] Found {len(self.image_paths)} test images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = tifffile.imread(path)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        image = image.astype(np.uint8)
        fname = os.path.basename(path)
        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = (
                torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            )
        return image, fname
