"""
train.py - 用全部 209 張訓練（不留 val split），強化 augmentation

Usage:
    python train.py \
        --data_root ./data/train \
        --output_dir ./checkpoints \
        --backbone resnet101 \
        --epochs 50 \
        --batch_size 1 \
        --lr 1e-4
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse, time, json, gc, random
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CellDataset, get_train_transform
from model import build_model, count_parameters


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, scaler, loader, device, epoch, print_freq=10):
    model.train()
    total_loss, comp_totals = 0.0, {}
    t0 = time.time()
    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(enabled=device.type == "cuda"):
            loss_dict = model(images, targets)
            losses = sum(v for v in loss_dict.values())

        if not torch.isfinite(losses):
            print(f"  [WARN] non-finite loss step {i}, skip")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()
        for k, v in loss_dict.items():
            comp_totals[k] = comp_totals.get(k, 0.0) + v.item()

        if (i + 1) % print_freq == 0:
            parts = "  ".join(f"{k}:{v.item():.3f}" for k, v in loss_dict.items())
            mem = (
                torch.cuda.memory_allocated(device) / 1e9
                if device.type == "cuda"
                else 0
            )
            print(
                f"  [{epoch}][{i+1}/{len(loader)}] loss={losses.item():.4f} "
                f"({parts})  GPU={mem:.1f}GB  {time.time()-t0:.1f}s"
            )
            t0 = time.time()

        if device.type == "cuda" and (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in comp_totals.items()}


def plot_losses(history, output_dir):
    epochs = list(range(1, len(history["train_total"]) + 1))
    components = [k for k in history if k != "train_total"]
    colors = ["#E74C3C", "#E67E22", "#2ECC71", "#9B59B6", "#1ABC9C"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_total"], label="Train", color="#E74C3C", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (full dataset)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    if components:
        fig, axes = plt.subplots(1, len(components), figsize=(4 * len(components), 4))
        if len(components) == 1:
            axes = [axes]
        for ax, comp, color in zip(axes, components, colors):
            ax.plot(epochs, history[comp], color=color, lw=2)
            ax.set_title(
                comp.replace("loss_", "").replace("_", " ").title(), fontsize=11
            )
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        fig.suptitle("Per-Component Train Losses", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "loss_components.png"), dpi=150)
        plt.close(fig)
    print(f"  📊 Curves saved → {output_dir}/")


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}  "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB"
        )

    # ── 用全部 209 張訓練，不留 val ───────────────────────────────────
    dataset = CellDataset(args.data_root, transforms=get_train_transform())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Training on ALL {len(dataset)} images (no val split)")

    model = build_model(num_classes=5, backbone_name=args.backbone, pretrained=True)
    model.to(device)
    count_parameters(model)

    backbone_p = [
        p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
    ]
    head_p = [
        p
        for n, p in model.named_parameters()
        if "backbone" not in n and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_p, "lr": args.lr * 0.1},
            {"params": head_p, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=device.type == "cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    hist_train, hist_comp = [], {}

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        tr_loss, tr_comp = train_one_epoch(
            model, optimizer, scaler, loader, device, epoch
        )
        scheduler.step()

        hist_train.append(tr_loss)
        for k, v in tr_comp.items():
            hist_comp.setdefault(k, []).append(v)

        print(f"  Train: {tr_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save every epoch (no val loss to compare, just keep all)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "train_loss": tr_loss,
                "backbone": args.backbone,
            },
            os.path.join(args.output_dir, "last_model.pth"),
        )

        # Save checkpoints
        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                os.path.join(args.output_dir, f"ckpt_ep{epoch}.pth"),
            )

        history = {"train_total": hist_train, **hist_comp}
        with open(os.path.join(args.output_dir, "loss_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        plot_losses(history, args.output_dir)

    print(f"\n✅ Done. Final train loss: {hist_train[-1]:.4f}")
    print("Checkpoints saved: last_model.pth + ckpt_ep{10,20,...}.pth")

    # --- 加入以下這段來強制清空資源 ---
    print("\nCleaning up GPU memory...")
    del model, loader, optimizer, scheduler  # 刪除佔用記憶體的變數
    gc.collect()  # 強制 Python 進行垃圾回收
    if device.type == "cuda":
        torch.cuda.empty_cache()  # 強制清空 PyTorch 的 CUDA 快取


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--backbone", type=str, default="resnet101", choices=["resnet50", "resnet101"]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
