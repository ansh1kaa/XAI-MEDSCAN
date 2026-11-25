"""Single-image prediction and CLI inference for XAI MedScan."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import BinaryIO, Optional, Union

from PIL import Image
import tensorflow as tf

from src.config import ProjectConfig, load_config
from src.data_utils import preprocess_image_for_inference
from src.gradcam import GradCAMResult, generate_gradcam_visuals
from src.model import load_trained_model


@dataclass
class PredictionResult:
    """Structured inference result for a single chest X-ray image."""

    predicted_class: str
    confidence: float
    normal_probability: float
    pneumonia_probability: float
    is_low_confidence: bool
    original_image: Image.Image
    gradcam: Optional[GradCAMResult] = None


def predict_image(
    model: tf.keras.Model,
    image_source: Union[Path, str, BinaryIO, Image.Image],
    config: ProjectConfig,
    include_gradcam: bool = True,
) -> PredictionResult:
    """Run MobileNetV2 prediction (and optional Grad-CAM) on one image."""
    original_image, batch = preprocess_image_for_inference(
        image_source, target_size=config.image_size
    )
    raw_pred = model.predict(batch, verbose=0)
    pneumonia_prob = float(raw_pred[0][0])
    pneumonia_prob = min(max(pneumonia_prob, 0.0), 1.0)
    normal_prob = 1.0 - pneumonia_prob

    if pneumonia_prob >= config.classification_threshold:
        predicted_class = "PNEUMONIA"
        confidence = pneumonia_prob
    else:
        predicted_class = "NORMAL"
        confidence = normal_prob

    is_low_confidence = confidence < config.low_confidence_threshold

    gradcam_res = None
    if include_gradcam:
        gradcam_res = generate_gradcam_visuals(
            model=model,
            image_source=original_image,
            config=config,
        )

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        normal_probability=normal_prob,
        pneumonia_probability=pneumonia_prob,
        is_low_confidence=is_low_confidence,
        original_image=original_image,
        gradcam=gradcam_res,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict NORMAL vs PNEUMONIA on a chest X-ray image with Grad-CAM."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to chest X-ray image")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument(
        "--save-overlay",
        type=str,
        default=None,
        help="Optional path to save the Grad-CAM overlay image",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image file not found: {image_path}")
        return 1

    try:
        model = load_trained_model(config.model_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    try:
        result = predict_image(model, image_path, config, include_gradcam=True)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("=== Prediction Result ===")
    print(f"Image            : {image_path}")
    print(f"Predicted Class  : {result.predicted_class}")
    print(f"Confidence       : {result.confidence * 100:.2f}%")
    print(f"NORMAL Prob      : {result.normal_probability * 100:.2f}%")
    print(f"PNEUMONIA Prob   : {result.pneumonia_probability * 100:.2f}%")
    if result.is_low_confidence:
        print("Warning          : Low-confidence prediction (< 70%).")

    if args.save_overlay and result.gradcam is not None:
        out_path = Path(args.save_overlay)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.gradcam.overlay_image.save(out_path)
        print(f"Grad-CAM overlay saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

