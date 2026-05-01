"""
plot_loss.py - Re-plot loss curves from a saved loss_history.json

Usage:
    python plot_loss.py --json ./checkpoints/loss_history.json --output_dir ./checkpoints
"""

import json
import argparse
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_losses(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(history["train_total"]) + 1))
    components = [k for k in history if k not in ("train_total", "val_total")]
    colors = ["#E74C3C", "#E67E22", "#2ECC71", "#9B59B6", "#1ABC9C", "#F39C12"]

    # ── Plot 1: Train vs Val total loss ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epochs,
        history["train_total"],
        label="Train Loss",
        color="#E74C3C",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.plot(
        epochs,
        history["val_total"],
        label="Val Loss",
        color="#3498DB",
        linewidth=2,
        marker="s",
        markersize=4,
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out1 = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── Plot 2: Per-component train losses ───────────────────────────
    if components:
        fig, axes = plt.subplots(1, len(components), figsize=(4 * len(components), 4))
        if len(components) == 1:
            axes = [axes]
        for ax, comp, color in zip(axes, components, colors):
            label = comp.replace("loss_", "").replace("_", " ").title()
            ax.plot(
                epochs,
                history[comp],
                color=color,
                linewidth=2,
                marker="o",
                markersize=3,
            )
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
        fig.suptitle("Per-Component Train Losses", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out2 = os.path.join(output_dir, "loss_components.png")
        fig.savefig(out2, dpi=150)
        plt.close(fig)
        print(f"Saved: {out2}")

    # ── Plot 3: Combined (total + all components) ────────────────────
    n = 1 + len(components)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    axes[0].plot(
        epochs, history["train_total"], label="Train", color="#E74C3C", linewidth=2
    )
    axes[0].plot(
        epochs, history["val_total"], label="Val", color="#3498DB", linewidth=2
    )
    axes[0].set_title("Total Loss", fontsize=11)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for ax, comp, color in zip(axes[1:], components, colors):
        label = comp.replace("loss_", "").replace("_", " ").title()
        ax.plot(epochs, history[comp], color=color, linewidth=2)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    fig.suptitle("All Loss Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out3 = os.path.join(output_dir, "loss_all.png")
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print(f"Saved: {out3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", type=str, required=True, help="Path to loss_history.json"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory to save plots"
    )
    args = parser.parse_args()

    with open(args.json) as f:
        history = json.load(f)

    print(
        f"Loaded history: epochs={len(history['train_total'])}, "
        f"keys={list(history.keys())}"
    )
    plot_losses(history, args.output_dir)
