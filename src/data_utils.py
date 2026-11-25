"""Dataset validation, image preprocessing, augmentation, and tf.data pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from src.config import ProjectConfig


SPLITS: Tuple[str, str, str] = ("train", "val", "test")


@dataclass
class DatasetValidationResult:
    """Structured report produced by `validate_dataset`."""

    data_dir: Path
    is_valid: bool
    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    valid_files: Dict[str, Dict[str, List[Path]]] = field(default_factory=dict)
    missing_dirs: List[str] = field(default_factory=list)
    invalid_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return sum(
            count
            for split_counts in self.counts.values()
            for count in split_counts.values()
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return a clean table of image counts across splits and classes."""
        rows = []
        for split in SPLITS:
            split_counts = self.counts.get(split, {})
            normal = split_counts.get("NORMAL", 0)
            pneumonia = split_counts.get("PNEUMONIA", 0)
            rows.append(
                {
                    "Split": split,
                    "NORMAL": normal,
                    "PNEUMONIA": pneumonia,
                    "Total": normal + pneumonia,
                }
            )
        df = pd.DataFrame(rows)
        totals = {
            "Split": "Total",
            "NORMAL": int(df["NORMAL"].sum()),
            "PNEUMONIA": int(df["PNEUMONIA"].sum()),
            "Total": int(df["Total"].sum()),
        }
        return pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    def format_summary(self) -> str:
        """Return human-readable summary of dataset validation."""
        lines = [f"Dataset directory: {self.data_dir}"]
        if self.missing_dirs:
            lines.append("Missing directories:")
            lines.extend(f"  - {d}" for d in self.missing_dirs)
        lines.append("\nImage counts:")
        lines.append(self.to_dataframe().to_string(index=False))
        if self.invalid_files:
            lines.append(f"\nSkipped invalid/corrupt images ({len(self.invalid_files)}):")
            lines.extend(f"  - {p}" for p in self.invalid_files[:10])
            if len(self.invalid_files) > 10:
                lines.append(f"  ... and {len(self.invalid_files) - 10} more.")
        if self.warnings:
            lines.append("\nWarnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def is_valid_image_file(path: Path) -> bool:
    """Verify an image file can be opened and decoded as RGB without error."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def validate_dataset(
    config: ProjectConfig,
    verify_images: bool = True,
) -> DatasetValidationResult:
    """Validate the Chest X-Ray directory structure and count valid images."""
    data_dir = Path(config.data_dir)
    counts: Dict[str, Dict[str, int]] = {
        split: {cls: 0 for cls in config.class_names} for split in SPLITS
    }
    valid_files: Dict[str, Dict[str, List[Path]]] = {
        split: {cls: [] for cls in config.class_names} for split in SPLITS
    }
    missing_dirs: List[str] = []
    invalid_files: List[str] = []
    warnings: List[str] = []

    if not data_dir.exists() or not data_dir.is_dir():
        missing_dirs.append(str(data_dir))
        return DatasetValidationResult(
            data_dir=data_dir,
            is_valid=False,
            counts=counts,
            valid_files=valid_files,
            missing_dirs=missing_dirs,
            invalid_files=invalid_files,
            warnings=["Dataset root directory does not exist."],
        )

    exts = {ext.lower() for ext in config.supported_extensions}

    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.exists() or not split_dir.is_dir():
            missing_dirs.append(str(split_dir))
            continue

        for cls in config.class_names:
            cls_dir = split_dir / cls
            if not cls_dir.exists() or not cls_dir.is_dir():
                missing_dirs.append(str(cls_dir))
                continue

            for item in sorted(cls_dir.iterdir()):
                if not item.is_file():
                    continue
                if item.suffix.lower() not in exts:
                    continue
                if verify_images and not is_valid_image_file(item):
                    invalid_files.append(str(item))
                    continue
                valid_files[split][cls].append(item)
            counts[split][cls] = len(valid_files[split][cls])

    empty_buckets = [
        f"{split}/{cls}"
        for split in SPLITS
        for cls in config.class_names
        if counts[split][cls] == 0 and str(data_dir / split / cls) not in missing_dirs
    ]
    if empty_buckets:
        warnings.append(
            "Empty class directories found: " + ", ".join(empty_buckets)
        )

    val_total = sum(counts["val"].values())
    if 0 < val_total <= 32:
        warnings.append(
            f"Validation split has only {val_total} images (standard Kaggle Chest X-Ray split)."
        )

    is_valid = len(missing_dirs) == 0 and all(
        counts[split][cls] > 0 for split in SPLITS for cls in config.class_names
    )

    return DatasetValidationResult(
        data_dir=data_dir,
        is_valid=is_valid,
        counts=counts,
        valid_files=valid_files,
        missing_dirs=missing_dirs,
        invalid_files=invalid_files,
        warnings=warnings,
    )


def load_rgb_image(source: Union[Path, str, BinaryIO, Image.Image]) -> Image.Image:
    """Load an image from disk, file-like stream, or PIL Image and convert to RGB."""
    try:
        if isinstance(source, Image.Image):
            img = source.copy()
        else:
            with Image.open(source) as opened:
                opened.load()
                img = opened.copy()
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid or unreadable image file: {exc}") from exc


def preprocess_pil_image(
    pil_image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize an RGB PIL image and apply MobileNetV2 preprocessing ([-1, 1])."""
    rgb = pil_image.convert("RGB")
    resized = rgb.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


def preprocess_image_for_inference(
    source: Union[Path, str, BinaryIO, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[Image.Image, np.ndarray]:
    """Return original RGB image and preprocessed `(1, H, W, 3)` batch tensor."""
    rgb_image = load_rgb_image(source)
    preprocessed = preprocess_pil_image(rgb_image, target_size=target_size)
    batch = np.expand_dims(preprocessed, axis=0)
    return rgb_image, batch


def build_augmentation_layer(config: ProjectConfig) -> tf.keras.Sequential:
    """Create training-only image augmentation pipeline."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=config.seed),
            tf.keras.layers.RandomRotation(
                factor=config.rotation_factor,
                fill_mode="nearest",
                seed=config.seed,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=config.zoom_factor,
                width_factor=config.zoom_factor,
                fill_mode="nearest",
                seed=config.seed,
            ),
        ],
        name="training_augmentation",
    )


def _collect_split_samples(
    validation: DatasetValidationResult,
    split: str,
    class_names: Tuple[str, str],
) -> Tuple[List[str], List[int]]:
    """Flatten file paths and integer labels (`NORMAL=0`, `PNEUMONIA=1`) for a split."""
    paths: List[str] = []
    labels: List[int] = []
    for idx, cls_name in enumerate(class_names):
        for file_path in validation.valid_files[split][cls_name]:
            paths.append(str(file_path))
            labels.append(idx)
    return paths, labels


def compute_balanced_class_weights(labels: List[int]) -> Dict[int, float]:
    """Compute balanced class weights to address train-set class imbalance."""
    unique_classes = np.array(sorted(set(labels)))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=np.array(labels),
    )
    return {int(cls): float(w) for cls, w in zip(unique_classes, weights)}


def create_tf_dataset(
    file_paths: List[str],
    labels: List[int],
    config: ProjectConfig,
    is_training: bool = False,
    augmentation_layer: Optional[tf.keras.Sequential] = None,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset with consistent MobileNetV2 preprocessing."""
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        ds = ds.shuffle(
            buffer_size=max(len(file_paths), 1),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )

    target_h, target_w = config.image_size

    def _load_and_resize(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [target_h, target_w], method="bilinear")
        image = tf.cast(image, tf.float32)
        return image, tf.cast(label, tf.float32)

    ds = ds.map(_load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.batch_size)

    if is_training and augmentation_layer is not None:
        ds = ds.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.map(
        lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def prepare_datasets(
    validation: DatasetValidationResult,
    config: ProjectConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[int, float], List[int]]:
    """Build train, val, and test tf.data.Datasets along with train class weights."""
    if not validation.is_valid:
        raise ValueError("Cannot prepare datasets: dataset validation failed.")

    train_paths, train_labels = _collect_split_samples(
        validation, "train", config.class_names
    )
    val_paths, val_labels = _collect_split_samples(
        validation, "val", config.class_names
    )
    test_paths, test_labels = _collect_split_samples(
        validation, "test", config.class_names
    )

    aug_layer = build_augmentation_layer(config)
    train_ds = create_tf_dataset(
        train_paths,
        train_labels,
        config=config,
        is_training=True,
        augmentation_layer=aug_layer,
    )
    val_ds = create_tf_dataset(
        val_paths,
        val_labels,
        config=config,
        is_training=False,
    )
    test_ds = create_tf_dataset(
        test_paths,
        test_labels,
        config=config,
        is_training=False,
    )
    class_weights = compute_balanced_class_weights(train_labels)
    return train_ds, val_ds, test_ds, class_weights, test_labels

