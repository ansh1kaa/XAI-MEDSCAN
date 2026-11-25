"""Grad-CAM (Gradient-weighted Class Activation Mapping) for MobileNetV2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Tuple, Union

import matplotlib
import numpy as np
from PIL import Image
import tensorflow as tf

from src.config import ProjectConfig
from src.data_utils import preprocess_image_for_inference


GRADCAM_DISCLAIMER = (
    "Highlighted regions represent image areas that influenced the model output. "
    "Grad-CAM is an approximate explanation and does not establish medical causality."
)


@dataclass
class GradCAMResult:
    """Outputs produced by Grad-CAM explanation."""

    original_image: Image.Image
    heatmap_array: np.ndarray
    heatmap_image: Image.Image
    overlay_image: Image.Image
    target_layer_name: str
    pneumonia_probability: float


def find_target_conv_layer(
    model: tf.keras.Model,
    preferred_name: str = "out_relu",
) -> str:
    """Identify the final suitable convolutional feature representation layer."""
    try:
        layer = model.get_layer(preferred_name)
        shape = getattr(layer, "output_shape", None)
        if shape is None and hasattr(layer, "output"):
            shape = tuple(layer.output.shape)
        if shape is not None and len(shape) == 4:
            return preferred_name
    except ValueError:
        pass

    for layer in reversed(model.layers):
        shape = None
        if hasattr(layer, "output"):
            try:
                shape = tuple(layer.output.shape)
            except Exception:
                shape = None
        if shape is None:
            shape = getattr(layer, "output_shape", None)
        if (
            shape is not None
            and len(shape) == 4
            and not isinstance(layer, tf.keras.layers.InputLayer)
        ):
            return layer.name

    raise ValueError(
        "Could not locate a 4D convolutional feature layer in the model for Grad-CAM."
    )


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    preprocessed_batch: np.ndarray,
    target_layer_name: str | None = None,
    preferred_layer: str = "out_relu",
) -> Tuple[np.ndarray, str, float]:
    """Compute normalized `[0, 1]` Grad-CAM heatmap for the PNEUMONIA score.

    Uses `tf.GradientTape` to compute gradients of the PNEUMONIA prediction score
    with respect to the target convolutional feature maps.
    """
    layer_name = target_layer_name or find_target_conv_layer(
        model, preferred_name=preferred_layer
    )
    target_layer = model.get_layer(layer_name)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    input_tensor = tf.cast(preprocessed_batch, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor, training=False)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        pneumonia_score = predictions[:, 0]

    grads = tape.gradient(pneumonia_score, conv_outputs)

    # Fallback for extreme sigmoid saturation where d(sigmoid(z))/dz underflows
    if grads is None or float(tf.reduce_max(tf.abs(grads))) < 1e-12:
        last_layer = model.layers[-1]
        if isinstance(last_layer, tf.keras.layers.Dense) and len(model.layers) >= 2:
            gap_layer = model.layers[-2]
            logit_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, gap_layer.output],
            )
            with tf.GradientTape() as tape_logit:
                conv_outputs, gap_out = logit_model(input_tensor, training=False)
                kernel, bias = last_layer.weights[0], last_layer.weights[1]
                logit_score = tf.squeeze(tf.matmul(gap_out, kernel) + bias, axis=-1)
            grads = tape_logit.gradient(logit_score, conv_outputs)

    if grads is None:
        raise RuntimeError("Failed to compute gradients for Grad-CAM.")

    # Channel-wise importance weights (global average pooling of gradients)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of activation maps
    conv_map = conv_outputs[0]
    heatmap = tf.squeeze(conv_map @ pooled_grads[..., tf.newaxis])

    # Apply ReLU and normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    if float(max_val) > 1e-12:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    return heatmap.numpy().astype(np.float32), layer_name, float(pneumonia_score[0])


def create_heatmap_and_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.40,
    colormap_name: str = "jet",
) -> Tuple[Image.Image, Image.Image, np.ndarray]:
    """Resize heatmap to match original image and blend into an overlay."""
    orig_rgb = original_image.convert("RGB")
    orig_w, orig_h = orig_rgb.size

    heatmap_uint8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0)
    heatmap_pil = Image.fromarray(heatmap_uint8, mode="L").resize(
        (orig_w, orig_h),
        resample=Image.Resampling.BILINEAR,
    )
    resized_norm = np.asarray(heatmap_pil, dtype=np.float32) / 255.0

    cmap = matplotlib.colormaps.get_cmap(colormap_name)
    colored_rgba = cmap(resized_norm)
    colored_rgb = np.uint8(colored_rgba[:, :, :3] * 255.0)
    heatmap_image = Image.fromarray(colored_rgb, mode="RGB")

    orig_arr = np.asarray(orig_rgb, dtype=np.float32)
    overlay_arr = (1.0 - alpha) * orig_arr + alpha * colored_rgb.astype(np.float32)
    overlay_arr = np.uint8(np.clip(overlay_arr, 0.0, 255.0))
    overlay_image = Image.fromarray(overlay_arr, mode="RGB")

    return heatmap_image, overlay_image, resized_norm


def generate_gradcam_visuals(
    model: tf.keras.Model,
    image_source: Union[str, BinaryIO, Image.Image],
    config: ProjectConfig,
) -> GradCAMResult:
    """End-to-end Grad-CAM generation for an input chest X-ray image."""
    original_image, batch = preprocess_image_for_inference(
        image_source, target_size=config.image_size
    )
    heatmap, layer_name, pneumonia_prob = compute_gradcam_heatmap(
        model=model,
        preprocessed_batch=batch,
        preferred_layer=config.preferred_gradcam_layer,
    )
    heatmap_image, overlay_image, resized_heatmap = create_heatmap_and_overlay(
        original_image=original_image,
        heatmap=heatmap,
        alpha=config.gradcam_alpha,
        colormap_name=config.gradcam_colormap,
    )
    return GradCAMResult(
        original_image=original_image,
        heatmap_array=resized_heatmap,
        heatmap_image=heatmap_image,
        overlay_image=overlay_image,
        target_layer_name=layer_name,
        pneumonia_probability=pneumonia_prob,
    )

