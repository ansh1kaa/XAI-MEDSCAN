"""MobileNetV2 transfer-learning architecture and model management helpers."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Iterator, List, Tuple

import numpy as np
import tensorflow as tf

from src.config import ProjectConfig


def set_global_seed(seed: int) -> None:
    """Set reproducible random seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def iter_all_layers(
    layer_or_model: tf.keras.layers.Layer,
) -> Iterator[tf.keras.layers.Layer]:
    """Recursively yield all child layers, including nested Models/Sequentials."""
    sublayers = getattr(layer_or_model, "layers", None)
    if not sublayers:
        return
    for sub in sublayers:
        yield sub
        yield from iter_all_layers(sub)


def count_batchnorm_layers(model: tf.keras.Model) -> Tuple[int, int, int]:
    """Recursively count BatchNormalization layers: (total, trainable, frozen)."""
    total = 0
    trainable = 0
    frozen = 0
    seen_ids = set()
    for layer in iter_all_layers(model):
        if id(layer) in seen_ids:
            continue
        seen_ids.add(id(layer))
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            total += 1
            if layer.trainable:
                trainable += 1
            else:
                frozen += 1
    return total, trainable, frozen


def verify_imagenet_weights_loaded(model: tf.keras.Model) -> None:
    """Verify ImageNet pretrained statistics are loaded (not random/default init).

    With `weights=None`, BatchNormalization moving_mean is all zeros and
    moving_variance is all ones. Pretrained ImageNet MobileNetV2 layers have
    non-trivial learned moving statistics.
    """
    bn_layers = [
        layer
        for layer in iter_all_layers(model)
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    if not bn_layers:
        raise RuntimeError(
            "Model verification failed: no BatchNormalization layers found in backbone."
        )

    max_mean_dev = 0.0
    max_var_dev = 0.0
    for bn in bn_layers:
        mean_arr = np.asarray(bn.moving_mean.numpy(), dtype=np.float32)
        var_arr = np.asarray(bn.moving_variance.numpy(), dtype=np.float32)
        max_mean_dev = max(max_mean_dev, float(np.max(np.abs(mean_arr))))
        max_var_dev = max(max_var_dev, float(np.max(np.abs(var_arr - 1.0))))

    if max_mean_dev < 1e-3 and max_var_dev < 1e-3:
        raise RuntimeError(
            "ImageNet pretrained weights do not appear to be loaded "
            "(BatchNormalization moving statistics remain at uninitialized 0/1 defaults). "
            "Ensure internet access or cached ImageNet weights before training."
        )


def compile_classifier(model: tf.keras.Model, learning_rate: float) -> None:
    """Compile binary classification model with Adam and binary cross-entropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def build_mobilenetv2_classifier(
    config: ProjectConfig,
    weights: str | None = "imagenet",
) -> Tuple[tf.keras.Model, List[str]]:
    """Build MobileNetV2 binary classifier with frozen backbone for initial training.

    Returns:
        (model, backbone_layer_names)
    """
    input_shape = (*config.image_size, config.channels)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )

    backbone_layer_names = [layer.name for layer in base_model.layers]
    for layer in base_model.layers:
        layer.trainable = False
        for nested in iter_all_layers(layer):
            nested.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(rate=config.dropout_rate, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="pneumonia_prediction",
    )(x)

    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=outputs,
        name="mobilenetv2_pneumonia_classifier",
    )
    compile_classifier(model, learning_rate=config.initial_learning_rate)
    return model, backbone_layer_names


def _get_backbone_layers(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    """Identify backbone layers preceding `global_avg_pool`."""
    backbone_layers: List[tf.keras.layers.Layer] = []
    for layer in model.layers:
        if layer.name == "global_avg_pool" or isinstance(
            layer, tf.keras.layers.GlobalAveragePooling2D
        ):
            break
        backbone_layers.append(layer)
    return backbone_layers


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    config: ProjectConfig,
) -> Tuple[int, int]:
    """Unfreeze only the last `fine_tune_last_n_layers` backbone layers.

    All BatchNormalization layers (top-level and nested) remain frozen to
    preserve pretrained population statistics during fine-tuning.

    Returns:
        (trainable_backbone_layers, total_frozen_bn_layers)
    """
    backbone_layers = _get_backbone_layers(model)
    total_backbone = len(backbone_layers)
    freeze_until = max(0, total_backbone - config.fine_tune_last_n_layers)

    for layer in backbone_layers[:freeze_until]:
        layer.trainable = False
        for nested in iter_all_layers(layer):
            nested.trainable = False

    trainable_tail = 0
    for layer in backbone_layers[freeze_until:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        elif getattr(layer, "layers", None):
            # Nested submodel (if present): unfreeze non-BN layers only
            layer.trainable = True
            for nested in iter_all_layers(layer):
                if isinstance(nested, tf.keras.layers.BatchNormalization):
                    nested.trainable = False
                else:
                    nested.trainable = True
            trainable_tail += 1
        else:
            layer.trainable = True
            trainable_tail += 1

    # Safety pass: enforce every BatchNormalization layer in the full model is frozen
    for layer in iter_all_layers(model):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    _, _, total_frozen_bn = count_batchnorm_layers(model)
    compile_classifier(model, learning_rate=config.fine_tune_learning_rate)
    return trainable_tail, total_frozen_bn


def load_trained_model(model_path: Path | str) -> tf.keras.Model:
    """Load a saved Keras model from disk with a clear missing-file error."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. "
            "Run `python -m src.train` first to train and save the model."
        )
    return tf.keras.models.load_model(path, compile=False)
