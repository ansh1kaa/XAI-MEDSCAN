"""Training pipeline for MobileNetV2 Chest X-Ray classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import tensorflow as tf

from src.config import ProjectConfig, load_config
from src.data_utils import prepare_datasets, validate_dataset
from src.evaluate import plot_training_curves
from src.model import (
    build_mobilenetv2_classifier,
    set_global_seed,
    unfreeze_for_fine_tuning,
    verify_imagenet_weights_loaded,
)


def _merge_histories(
    h1: Dict[str, List[float]],
    h2: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """Concatenate metrics from Phase 1 (head) and Phase 2 (fine-tuning)."""
    merged: Dict[str, List[float]] = {}
    all_keys = sorted(set(h1.keys()) | set(h2.keys()))
    for key in all_keys:
        vals1 = [float(v) for v in h1.get(key, [])]
        vals2 = [float(v) for v in h2.get(key, [])]
        merged[key] = vals1 + vals2
    return merged


def _build_callbacks(
    config: ProjectConfig,
    checkpoint_path: Path,
) -> List[tf.keras.callbacks.Callback]:
    """Create EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau callbacks."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=1,
        ),
    ]


def train_model(
    config: ProjectConfig,
    verify_images: bool = True,
) -> bool:
    """Execute full two-stage transfer learning and save best model + history."""
    set_global_seed(config.seed)
    config.ensure_directories()

    print("=== Step 1: Validating Dataset ===")
    validation = validate_dataset(config, verify_images=verify_images)
    print(validation.format_summary())

    if not validation.is_valid:
        print(
            "\n[ERROR] Dataset is incomplete or missing. "
            "Please download the Chest X-Ray Images (Pneumonia) dataset from:\n"
            "  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n"
            f"and extract it to: {config.data_dir}"
        )
        return False

    print("\n=== Step 2: Preparing Data Pipelines ===")
    train_ds, val_ds, _test_ds, class_weights, _ = prepare_datasets(
        validation, config
    )
    print(f"Computed balanced class weights: {class_weights}")

    print("\n=== Step 3: Building MobileNetV2 Model (Frozen Backbone, ImageNet Weights) ===")
    try:
        model, backbone_layers = build_mobilenetv2_classifier(config, weights="imagenet")
        verify_imagenet_weights_loaded(model)
    except Exception as exc:
        print(
            "\n[ERROR] Could not load and verify ImageNet pretrained weights for MobileNetV2: "
            f"{exc}\nTraining aborted to prevent training from uninitialized random weights."
        )
        return False
    print(f"Total backbone layers: {len(backbone_layers)} (ImageNet weights verified, all frozen for Phase 1)")

    print(
        f"\n=== Step 4: Phase 1 Training (Head Only, {config.initial_epochs} epochs, lr={config.initial_learning_rate}) ==="
    )
    phase1_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.initial_epochs,
        class_weight=class_weights,
        callbacks=_build_callbacks(config, config.model_path),
        verbose=1,
    )

    h1_dict = {k: [float(x) for x in v] for k, v in phase1_history.history.items()}
    fine_tune_start_epoch = len(h1_dict.get("loss", []))
    h2_dict: Dict[str, List[float]] = {}

    if config.fine_tune_epochs > 0:
        if config.model_path.exists():
            model = tf.keras.models.load_model(config.model_path)

        print(
            f"\n=== Step 5: Phase 2 Fine-Tuning (Last {config.fine_tune_last_n_layers} Backbone Layers, "
            f"BN Frozen, {config.fine_tune_epochs} epochs, lr={config.fine_tune_learning_rate}) ==="
        )
        trainable_tail, frozen_bn_tail = unfreeze_for_fine_tuning(model, config)
        print(
            f"Fine-tuning tail: {trainable_tail} trainable layers, "
            f"{frozen_bn_tail} BatchNormalization layers kept frozen."
        )

        total_target_epochs = fine_tune_start_epoch + config.fine_tune_epochs
        phase2_history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=fine_tune_start_epoch,
            epochs=total_target_epochs,
            class_weight=class_weights,
            callbacks=_build_callbacks(config, config.model_path),
            verbose=1,
        )
        h2_dict = {k: [float(x) for x in v] for k, v in phase2_history.history.items()}

    # Ensure model file exists even if EarlyStopping didn't trigger checkpointing
    if not config.model_path.exists():
        model.save(config.model_path)

    merged_history = _merge_histories(h1_dict, h2_dict)
    history_payload = {
        "fine_tune_start_epoch": fine_tune_start_epoch,
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "dataset_counts": validation.counts,
        "history": merged_history,
    }
    with config.history_path.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    plot_training_curves(
        merged_history,
        accuracy_path=config.accuracy_plot_path,
        loss_path=config.loss_plot_path,
        fine_tune_start_epoch=fine_tune_start_epoch,
    )

    print(f"\nTraining complete. Best model saved to: {config.model_path}")
    print(f"Training history saved to: {config.history_path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 Chest X-Ray Pneumonia Classifier."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--epochs", type=int, default=None, help="Override Phase 1 epochs")
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=None, help="Override Phase 2 epochs"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip per-file Pillow integrity check during dataset scan",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config.initial_epochs = args.epochs
    if args.fine_tune_epochs is not None:
        config.fine_tune_epochs = args.fine_tune_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    success = train_model(config, verify_images=not args.skip_verify)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

