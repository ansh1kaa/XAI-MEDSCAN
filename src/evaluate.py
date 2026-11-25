"""Model evaluation on the held-out test split and plot generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import ProjectConfig, load_config
from src.data_utils import create_tf_dataset, validate_dataset
from src.model import load_trained_model


def plot_training_curves(
    history: Dict[str, List[float]],
    accuracy_path: Path,
    loss_path: Path,
    fine_tune_start_epoch: Optional[int] = None,
) -> None:
    """Save training & validation accuracy and loss curves."""
    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    if acc and val_acc:
        epochs = list(range(1, len(acc) + 1))
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        ax.plot(epochs, acc, marker="o", linewidth=2, color="#1D6FA5", label="Train Accuracy")
        ax.plot(epochs, val_acc, marker="s", linewidth=2, color="#2CA58D", label="Val Accuracy")
        if fine_tune_start_epoch and 0 < fine_tune_start_epoch < len(epochs):
            ax.axvline(
                x=fine_tune_start_epoch + 0.5,
                color="#64748B",
                linestyle="--",
                linewidth=1.2,
                label="Fine-Tuning Starts",
            )
        ax.set_title("Training and Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(accuracy_path)
        plt.close(fig)

    if loss and val_loss:
        epochs = list(range(1, len(loss) + 1))
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        ax.plot(epochs, loss, marker="o", linewidth=2, color="#1D6FA5", label="Train Loss")
        ax.plot(epochs, val_loss, marker="s", linewidth=2, color="#E07A5F", label="Val Loss")
        if fine_tune_start_epoch and 0 < fine_tune_start_epoch < len(epochs):
            ax.axvline(
                x=fine_tune_start_epoch + 0.5,
                color="#64748B",
                linestyle="--",
                linewidth=1.2,
                label="Fine-Tuning Starts",
            )
        ax.set_title("Training and Validation Loss", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary Cross-Entropy Loss")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(loss_path)
        plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: tuple[str, str],
    output_path: Path,
) -> None:
    """Plot and save a clean confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Actual Class",
        xlabel="Predicted Class",
    )
    ax.set_title(
        "Test Set Confusion Matrix\n(Positive Class: PNEUMONIA)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    total = cm.sum() if cm.sum() > 0 else 1
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100.0 * cm[i, j] / total
            color = "white" if cm[i, j] > thresh else "#0F172A"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: Optional[float],
    output_path: Path,
) -> None:
    """Plot and save the ROC curve for positive class PNEUMONIA."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    if len(np.unique(y_true)) == 2 and roc_auc is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
        ax.plot(
            fpr,
            tpr,
            color="#1D6FA5",
            linewidth=2.4,
            label=f"MobileNetV2 (ROC-AUC = {roc_auc:.4f})",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "ROC Curve requires both NORMAL and PNEUMONIA samples.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title(
        "ROC Curve (Positive Class: PNEUMONIA)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_model(
    config: ProjectConfig,
    verify_images: bool = True,
) -> Optional[Dict[str, object]]:
    """Evaluate trained model on the held-out test set and save all artifacts."""
    config.ensure_directories()

    if not config.model_path.exists():
        print(f"[ERROR] Trained model not found at: {config.model_path}")
        print("Train the model first with: python -m src.train")
        return None

    validation = validate_dataset(config, verify_images=verify_images)
    test_counts = validation.counts.get("test", {})
    if test_counts.get("NORMAL", 0) == 0 or test_counts.get("PNEUMONIA", 0) == 0:
        print("[ERROR] Test dataset split is missing or empty.")
        print(validation.format_summary())
        return None

    test_paths: List[str] = []
    test_labels: List[int] = []
    for idx, cls_name in enumerate(config.class_names):
        for path in validation.valid_files["test"][cls_name]:
            test_paths.append(str(path))
            test_labels.append(idx)

    test_ds = create_tf_dataset(
        test_paths,
        test_labels,
        config=config,
        is_training=False,
    )

    print(f"Loading model from: {config.model_path}")
    model = load_trained_model(config.model_path)

    print(f"Evaluating on {len(test_labels)} test images...")
    y_true = np.asarray(test_labels, dtype=int)
    y_prob = model.predict(test_ds, verbose=1).reshape(-1)
    y_pred = (y_prob >= config.classification_threshold).astype(int)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    roc_auc: Optional[float] = (
        float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report_txt = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=list(config.class_names),
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=list(config.class_names),
        output_dict=True,
        zero_division=0,
    )

    # Save plots
    plot_confusion_matrix(cm, config.class_names, config.confusion_matrix_plot_path)
    plot_roc_curve(y_true, y_prob, roc_auc, config.roc_curve_plot_path)

    if config.history_path.exists():
        with config.history_path.open("r", encoding="utf-8") as f:
            hist_payload = json.load(f)
        plot_training_curves(
            hist_payload.get("history", {}),
            accuracy_path=config.accuracy_plot_path,
            loss_path=config.loss_plot_path,
            fine_tune_start_epoch=hist_payload.get("fine_tune_start_epoch"),
        )

    # Save text & JSON reports
    with config.report_txt_path.open("w", encoding="utf-8") as f:
        f.write("Classification Report (Positive Class: PNEUMONIA = 1)\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_txt)

    with config.report_json_path.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    metrics_payload: Dict[str, object] = {
        "positive_class": config.positive_class,
        "positive_class_index": config.positive_class_index,
        "classification_threshold": config.classification_threshold,
        "num_test_samples": int(len(y_true)),
        "test_class_counts": test_counts,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(config.class_names),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    with config.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n=== Held-Out Test Metrics (Positive Class: PNEUMONIA) ===")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC   : N/A")
    print("\nClassification Report:\n")
    print(report_txt)
    print(f"Saved evaluation results to: {config.results_dir}")
    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained MobileNetV2 model on test split."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip per-file Pillow integrity check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    res = evaluate_model(config, verify_images=not args.skip_verify)
    return 0 if res is not None else 1


if __name__ == "__main__":
    sys.exit(main())

