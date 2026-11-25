"""Generate system architecture and model architecture diagrams in assets/."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ASSETS_DIR = Path(__file__).resolve().parent


def _draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    facecolor: str = "#EFF6FF",
    edgecolor: str = "#1D6FA5",
    title_color: str = "#0F3D5E",
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=title_color,
        )
        ax.text(
            x + w / 2,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#334155",
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=title_color,
        )


def _draw_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.5", color="#1D6FA5", lw=1.8),
    )


def generate_system_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=180)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    ax.text(
        5.0,
        5.15,
        "XAI MEDSCAN — System Architecture",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#0F3D5E",
    )

    # Top pipeline: Offline Training & Evaluation
    ax.text(0.3, 4.55, "1. Offline Training & Evaluation Pipeline", fontsize=10, fontweight="bold", color="#3B6998")
    _draw_box(ax, 0.3, 3.3, 1.9, 1.0, "Chest X-Ray Dataset", "train / val / test\nNORMAL & PNEUMONIA", "#F8FAFC", "#64748B")
    _draw_box(ax, 2.7, 3.3, 2.1, 1.0, "Data Validation &\nAugmentation", "RGB, 224x224, Flip,\nRot, Zoom (train only)")
    _draw_box(ax, 5.3, 3.3, 2.1, 1.0, "2-Stage Training", "Frozen Base + Fine-Tune\nLast 30 Layers (BN Frozen)")
    _draw_box(ax, 7.9, 3.3, 1.8, 1.0, "Saved Artifacts", ".keras Model, Metrics,\nROC & Confusion Matrix", "#ECFDF5", "#059669", "#065F46")

    _draw_arrow(ax, 2.22, 3.8, 2.68, 3.8)
    _draw_arrow(ax, 4.82, 3.8, 5.28, 3.8)
    _draw_arrow(ax, 7.42, 3.8, 7.88, 3.8)

    # Bottom pipeline: Interactive Web App & XAI
    ax.text(0.3, 2.55, "2. Interactive Inference & Grad-CAM Explainability (Streamlit UI)", fontsize=10, fontweight="bold", color="#3B6998")
    _draw_box(ax, 0.3, 1.1, 1.9, 1.05, "User Upload (UI)", "JPG / JPEG / PNG\nChest X-Ray Input", "#F8FAFC", "#64748B")
    _draw_box(ax, 2.7, 1.1, 2.1, 1.05, "Consistent Preprocessing", "RGB -> 224x224\nMobileNetV2 [-1, 1]")
    _draw_box(ax, 5.3, 1.1, 2.1, 1.05, "MobileNetV2 + Grad-CAM", "Sigmoid Probability &\nGradientTape Heatmap")
    _draw_box(ax, 7.9, 1.1, 1.8, 1.05, "XAI MEDSCAN View", "Class, Confidence,\nHeatmap & Overlay", "#EFF6FF", "#1D6FA5")

    _draw_arrow(ax, 2.22, 1.62, 2.68, 1.62)
    _draw_arrow(ax, 4.82, 1.62, 5.28, 1.62)
    _draw_arrow(ax, 7.42, 1.62, 7.88, 1.62)

    # Model link from top to bottom
    _draw_arrow(ax, 8.8, 3.28, 6.35, 2.18)

    ax.text(
        5.0,
        0.35,
        "Educational Academic Prototype Only — Not Intended for Clinical Diagnosis",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        color="#92400E",
    )

    out_path = ASSETS_DIR / "system_architecture.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_model_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 7.6), dpi=180)
    ax.set_xlim(0, 6.8)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    ax.text(
        3.0,
        8.2,
        "Model Architecture (Transfer Learning)",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color="#0F3D5E",
    )

    blocks = [
        ("Input Chest X-Ray Image", "224 × 224 × 3 (RGB)", "#F8FAFC", "#64748B"),
        ("MobileNetV2 Preprocessing", "Scale pixels to [-1, 1]", "#EFF6FF", "#1D6FA5"),
        ("MobileNetV2 Backbone", "ImageNet Pretrained (include_top=False)\nFinal Conv Map: out_relu (7 × 7 × 1280)", "#E0F2FE", "#0284C7"),
        ("Global Average Pooling 2D", "Spatial 7×7×1280 → 1280 features", "#EFF6FF", "#1D6FA5"),
        ("Dropout Layer", "Rate = 0.35 (Regularization)", "#EFF6FF", "#1D6FA5"),
        ("Dense Output Layer", "1 Neuron, Sigmoid Activation", "#ECFDF5", "#059669"),
        ("Binary Prediction", "NORMAL (0) vs PNEUMONIA (1)", "#FEF2F2", "#DC2626"),
    ]

    y_top = 7.1
    step = 1.08
    box_w = 4.4
    box_h = 0.72
    x_left = 0.6

    for idx, (title, sub, fc, ec) in enumerate(blocks):
        y = y_top - idx * step
        _draw_box(ax, x_left, y, box_w, box_h, title, sub, facecolor=fc, edgecolor=ec)
        if idx < len(blocks) - 1:
            _draw_arrow(ax, x_left + box_w / 2, y - 0.02, x_left + box_w / 2, y - (step - box_h) + 0.02)

    # Grad-CAM annotation from Dense Output Layer back to MobileNetV2 out_relu
    ax.annotate(
        "Grad-CAM\nGradientTape",
        xy=(x_left + box_w + 0.06, y_top - 2 * step + box_h / 2),
        xytext=(5.95, y_top - 5 * step + box_h / 2),
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="#B45309",
        arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.8, connectionstyle="arc3,rad=0.25"),
    )

    out_path = ASSETS_DIR / "model_architecture.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    s = generate_system_architecture()
    m = generate_model_architecture()
    print(f"Generated: {s}")
    print(f"Generated: {m}")

