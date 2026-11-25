"""Automated verification suite for XAI MEDSCAN.

Produces a structured PASS / FAIL / NOT TESTED report across six checks,
strictly separating synthetic pipeline checks from real medical classification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Dict, List

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CheckResult:
    check_id: int
    name: str
    status: str  # "PASS", "FAIL", or "NOT TESTED"
    details: str


def check_1_structure_and_imports() -> CheckResult:
    """Check 1: Required project files and Python module imports."""
    required_paths = [
        PROJECT_ROOT / "app" / "streamlit_app.py",
        PROJECT_ROOT / "src" / "__init__.py",
        PROJECT_ROOT / "src" / "config.py",
        PROJECT_ROOT / "src" / "data_utils.py",
        PROJECT_ROOT / "src" / "model.py",
        PROJECT_ROOT / "src" / "train.py",
        PROJECT_ROOT / "src" / "evaluate.py",
        PROJECT_ROOT / "src" / "gradcam.py",
        PROJECT_ROOT / "src" / "predict.py",
        PROJECT_ROOT / "data" / "README.md",
        PROJECT_ROOT / "models" / ".gitkeep",
        PROJECT_ROOT / "results" / ".gitkeep",
        PROJECT_ROOT / "assets" / "system_architecture.png",
        PROJECT_ROOT / "assets" / "model_architecture.png",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / ".gitignore",
        PROJECT_ROOT / "run.bat",
    ]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in required_paths if not p.exists()]
    if missing:
        return CheckResult(1, "Project Structure & Imports", "FAIL", f"Missing files: {missing}")

    try:
        import src.config  # noqa: F401
        import src.data_utils  # noqa: F401
        import src.model  # noqa: F401
        import src.train  # noqa: F401
        import src.evaluate  # noqa: F401
        import src.gradcam  # noqa: F401
        import src.predict  # noqa: F401
        import streamlit  # noqa: F401
    except Exception as exc:
        return CheckResult(1, "Project Structure & Imports", "FAIL", f"Import error: {exc}")

    return CheckResult(
        1,
        "Project Structure & Imports",
        "PASS",
        f"All {len(required_paths)} required files and core modules verified.",
    )


def check_2_dataset_validation() -> CheckResult:
    """Check 2: Dataset validation handles missing dirs and corrupt files safely."""
    from src.config import load_config
    from src.data_utils import validate_dataset

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cfg = load_config()
            cfg.data_dir = tmp_dir / "missing_chest_xray"
            res_missing = validate_dataset(cfg)
            assert not res_missing.is_valid
            assert len(res_missing.missing_dirs) > 0

            for split in ("train", "val", "test"):
                for cls in ("NORMAL", "PNEUMONIA"):
                    cls_dir = tmp_dir / "mini_xray" / split / cls
                    cls_dir.mkdir(parents=True, exist_ok=True)
                    img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
                    img.save(cls_dir / "sample.png")
            (tmp_dir / "mini_xray" / "train" / "NORMAL" / "corrupt.jpg").write_bytes(
                b"not-an-image"
            )

            cfg.data_dir = tmp_dir / "mini_xray"
            res_valid = validate_dataset(cfg, verify_images=True)
            assert res_valid.is_valid
            assert res_valid.counts["train"]["NORMAL"] == 1
            assert len(res_valid.invalid_files) == 1
    except Exception as exc:
        return CheckResult(2, "Dataset Validation Logic", "FAIL", str(exc))

    return CheckResult(
        2,
        "Dataset Validation Logic",
        "PASS",
        "Missing directory detection and corrupt-image skipping verified.",
    )


def check_3_model_architecture_bn_and_imagenet() -> CheckResult:
    """Check 3: Model shapes, recursive BN freeze (incl. nested MobileNetV2), and ImageNet guard."""
    import tensorflow as tf
    from src.config import load_config
    from src.model import (
        build_mobilenetv2_classifier,
        count_batchnorm_layers,
        iter_all_layers,
        set_global_seed,
        unfreeze_for_fine_tuning,
        verify_imagenet_weights_loaded,
    )

    try:
        cfg = load_config()
        set_global_seed(cfg.seed)

        # 3a. Verify the ImageNet guard rejects random-weight initialization
        random_model, _ = build_mobilenetv2_classifier(cfg, weights=None)
        rejected_random = False
        try:
            verify_imagenet_weights_loaded(random_model)
        except RuntimeError:
            rejected_random = True
        assert rejected_random, "verify_imagenet_weights_loaded must reject weights=None models."

        # 3b. Verify nested backbone wrapping also keeps all nested BN layers frozen
        nested_base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights=None
        )
        nested_inp = tf.keras.Input(shape=(224, 224, 3))
        nx = nested_base(nested_inp)
        nx = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(nx)
        nx = tf.keras.layers.Dropout(cfg.dropout_rate)(nx)
        nout = tf.keras.layers.Dense(1, activation="sigmoid")(nx)
        nested_model = tf.keras.Model(nested_inp, nout)
        unfreeze_for_fine_tuning(nested_model, cfg)
        n_total_bn, n_train_bn, n_frozen_bn = count_batchnorm_layers(nested_model)
        assert n_total_bn > 0 and n_train_bn == 0 and n_frozen_bn == n_total_bn, (
            f"Nested MobileNetV2 BN freeze failed: total={n_total_bn}, trainable={n_train_bn}"
        )

        # 3c. Verify actual MobileNetV2 classifier with ImageNet pretrained weights
        try:
            model, backbone_names = build_mobilenetv2_classifier(cfg, weights="imagenet")
            verify_imagenet_weights_loaded(model)
        except Exception as exc:
            return CheckResult(
                3,
                "Model, ImageNet Weights & Recursive BN Freeze",
                "FAIL",
                f"ImageNet pretrained weights could not be loaded/verified: {exc}",
            )

        assert tuple(model.input_shape) == (None, 224, 224, 3)
        assert tuple(model.output_shape) == (None, 1)
        assert len(backbone_names) > 100

        trainable_tail, frozen_bn_count = unfreeze_for_fine_tuning(model, cfg)
        total_bn, trainable_bn, frozen_bn = count_batchnorm_layers(model)
        assert trainable_tail > 0
        assert total_bn > 0 and trainable_bn == 0 and frozen_bn == frozen_bn_count
        for layer in iter_all_layers(model):
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                assert not layer.trainable

    except Exception as exc:
        return CheckResult(3, "Model, ImageNet Weights & Recursive BN Freeze", "FAIL", str(exc))

    return CheckResult(
        3,
        "Model, ImageNet Weights & Recursive BN Freeze",
        "PASS",
        (
            f"Input (224,224,3)->(1,) verified; ImageNet pretrained weights verified; "
            f"all {frozen_bn} top-level & nested MobileNetV2 BatchNorm layers remain frozen."
        ),
    )


def check_4_synthetic_pipeline_and_gradcam() -> CheckResult:
    """Check 4: Synthetic-image preprocessing, tensor shapes, and Grad-CAM mechanics ONLY.

    NOTE: Does NOT evaluate or validate real medical classification accuracy.
    """
    from src.config import load_config
    from src.data_utils import preprocess_image_for_inference
    from src.gradcam import generate_gradcam_visuals
    from src.model import build_mobilenetv2_classifier, set_global_seed

    try:
        cfg = load_config()
        set_global_seed(cfg.seed)
        # Explicitly build an untrained topology for mechanics-only pipeline verification
        mechanics_model, _ = build_mobilenetv2_classifier(cfg, weights=None)

        synthetic_img = Image.fromarray(
            np.random.RandomState(42).randint(0, 255, (300, 320, 3), dtype=np.uint8)
        )
        orig, batch = preprocess_image_for_inference(synthetic_img, target_size=cfg.image_size)
        assert orig.mode == "RGB"
        assert batch.shape == (1, 224, 224, 3)
        assert -1.0001 <= float(batch.min()) <= float(batch.max()) <= 1.0001

        raw_score = float(mechanics_model.predict(batch, verbose=0)[0][0])
        assert 0.0 <= raw_score <= 1.0

        gcam = generate_gradcam_visuals(mechanics_model, synthetic_img, cfg)
        assert gcam.target_layer_name == "out_relu"
        assert gcam.overlay_image.size == synthetic_img.size
        assert gcam.heatmap_image.size == synthetic_img.size
        assert 0.0 <= float(gcam.heatmap_array.min()) <= float(gcam.heatmap_array.max()) <= 1.0
    except Exception as exc:
        return CheckResult(4, "Synthetic Pipeline & Grad-CAM Mechanics", "FAIL", str(exc))

    return CheckResult(
        4,
        "Synthetic Pipeline & Grad-CAM Mechanics",
        "PASS",
        "Preprocessing [-1,1], tensor shapes, and Grad-CAM overlay mechanics verified on synthetic image (not a medical prediction check).",
    )


def check_5_streamlit_dashboard() -> CheckResult:
    """Check 5: All 5 Streamlit dashboard pages render without runtime exceptions."""
    try:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=30)
        at.run()
        assert not at.exception, f"Home page exception: {at.exception}"

        for page_name in [
            "Image Prediction",
            "Explainability (Grad-CAM)",
            "Model Performance",
            "About Project",
        ]:
            at.sidebar.radio[0].set_value(page_name).run()
            assert not at.exception, f"{page_name} exception: {at.exception}"
    except Exception as exc:
        return CheckResult(5, "Streamlit Dashboard Multi-Page Rendering", "FAIL", str(exc))

    return CheckResult(
        5,
        "Streamlit Dashboard Multi-Page Rendering",
        "PASS",
        "All 5 Streamlit UI sections rendered cleanly with zero exceptions.",
    )


def check_6_real_medical_classification() -> tuple[CheckResult, Dict[str, object]]:
    """Check 6: Real Chest X-Ray dataset, trained model, and real classification check.

    Never falls back to random weights or synthetic images. Reports NOT TESTED
    when the real dataset or trained model is unavailable.
    """
    from src.config import load_config
    from src.data_utils import validate_dataset
    from src.model import load_trained_model
    from src.predict import predict_image

    cfg = load_config()
    ds = validate_dataset(cfg, verify_images=False)
    model_exists = cfg.model_path.exists()
    metrics_exists = cfg.metrics_path.exists()

    env_status: Dict[str, object] = {
        "real_dataset_available": ds.is_valid,
        "real_dataset_path": str(cfg.data_dir),
        "real_dataset_total_images": ds.total_images,
        "trained_model_available": model_exists,
        "trained_model_path": str(cfg.model_path),
        "test_metrics_available": metrics_exists,
        "test_metrics_path": str(cfg.metrics_path),
    }

    if not ds.is_valid or not model_exists:
        missing_parts = []
        if not ds.is_valid:
            missing_parts.append(f"real dataset missing at '{cfg.data_dir}'")
        if not model_exists:
            missing_parts.append(f"trained model missing at '{cfg.model_path}'")
        return (
            CheckResult(
                6,
                "Real Medical Image Classification",
                "NOT TESTED",
                "Cannot test real medical classification: " + "; ".join(missing_parts) + ".",
            ),
            env_status,
        )

    try:
        trained_model = load_trained_model(cfg.model_path)
        sample_paths: List[Path] = []
        for cls_name in cfg.class_names:
            cls_files = ds.valid_files["test"].get(cls_name, [])
            if cls_files:
                sample_paths.append(cls_files[0])
        if not sample_paths:
            return (
                CheckResult(
                    6,
                    "Real Medical Image Classification",
                    "FAIL",
                    "Real dataset valid flag set, but no test image paths were found.",
                ),
                env_status,
            )

        for img_path in sample_paths:
            res = predict_image(trained_model, img_path, cfg, include_gradcam=True)
            assert res.predicted_class in cfg.class_names
            assert res.gradcam is not None

        metrics_note = (
            f"test_metrics.json present at '{cfg.metrics_path}'"
            if metrics_exists
            else "test_metrics.json not yet generated (run `python -m src.evaluate`)"
        )
        return (
            CheckResult(
                6,
                "Real Medical Image Classification",
                "PASS",
                f"Trained model verified on {len(sample_paths)} real chest X-ray test sample(s) with Grad-CAM; {metrics_note}.",
            ),
            env_status,
        )
    except Exception as exc:
        return (
            CheckResult(6, "Real Medical Image Classification", "FAIL", str(exc)),
            env_status,
        )


def format_report(results: List[CheckResult], env_status: Dict[str, object]) -> str:
    lines = [
        "=" * 78,
        "XAI MEDSCAN — AUTOMATED VERIFICATION REPORT",
        "=" * 78,
        f"Real Dataset Available  : {'YES' if env_status['real_dataset_available'] else 'NO'} ({env_status['real_dataset_path']}, images={env_status['real_dataset_total_images']})",
        f"Trained Model Available : {'YES' if env_status['trained_model_available'] else 'NO'} ({env_status['trained_model_path']})",
        f"Test Metrics Available  : {'YES' if env_status['test_metrics_available'] else 'NO'} ({env_status['test_metrics_path']})",
        "-" * 78,
    ]
    for r in results:
        lines.append(f"[{r.status:<10}] Check {r.check_id}: {r.name}")
        lines.append(f"             {r.details}")
    lines.append("-" * 78)
    counts = {
        s: sum(1 for r in results if r.status == s)
        for s in ("PASS", "FAIL", "NOT TESTED")
    }
    lines.append(
        f"Summary: PASS={counts['PASS']} | FAIL={counts['FAIL']} | NOT TESTED={counts['NOT TESTED']}"
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    results: List[CheckResult] = [
        check_1_structure_and_imports(),
        check_2_dataset_validation(),
        check_3_model_architecture_bn_and_imagenet(),
        check_4_synthetic_pipeline_and_gradcam(),
        check_5_streamlit_dashboard(),
    ]
    check_6_res, env_status = check_6_real_medical_classification()
    results.append(check_6_res)

    report_text = format_report(results, env_status)
    print("\n" + report_text)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "verification_report.txt").write_text(report_text + "\n", encoding="utf-8")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": env_status,
        "checks": [asdict(r) for r in results],
    }
    (results_dir / "verification_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    has_failure = any(r.status == "FAIL" for r in results)
    return 1 if has_failure else 0


if __name__ == "__main__":
    sys.exit(main())
