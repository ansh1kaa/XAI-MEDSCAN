"""XAI MEDSCAN — Explainable Medical Image Classification Streamlit Dashboard."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from PIL import Image
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ProjectConfig, load_config  # noqa: E402
from src.data_utils import DatasetValidationResult, validate_dataset  # noqa: E402
from src.gradcam import GRADCAM_DISCLAIMER  # noqa: E402
from src.model import load_trained_model  # noqa: E402
from src.predict import PredictionResult, predict_image  # noqa: E402


MEDICAL_DISCLAIMER = (
    "Academic Demonstration Only: XAI MEDSCAN is an educational B.Tech Machine "
    "Learning PBL prototype and is NOT a certified medical diagnostic tool. Do not "
    "use its predictions or Grad-CAM heatmaps for clinical decision-making."
)


st.set_page_config(
    page_title="XAI MEDSCAN | Explainable Medical Image Classification",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    """Apply clean healthcare-inspired white and soft blue styling."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }
        .xai-header {
            background: linear-gradient(135deg, #F0F7FF 0%, #FFFFFF 100%);
            border: 1px solid #D6E6F5;
            border-left: 5px solid #1D6FA5;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
        }
        .xai-title {
            color: #0F3D5E;
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin: 0;
        }
        .xai-subtitle {
            color: #3B6998;
            font-size: 1rem;
            font-weight: 500;
            margin-top: 0.25rem;
            margin-bottom: 0;
        }
        .xai-card {
            background-color: #FFFFFF;
            border: 1px solid #DCE7F3;
            border-radius: 12px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }
        .xai-card h4 {
            color: #0F3D5E;
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 1.05rem;
        }
        .xai-disclaimer {
            background-color: #FFF8E6;
            border: 1px solid #F6D88C;
            border-left: 5px solid #D97706;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            color: #78350F;
            font-size: 0.92rem;
            margin-bottom: 1.1rem;
        }
        .result-badge-normal {
            background-color: #ECFDF5;
            border: 1px solid #A7F3D0;
            border-left: 5px solid #059669;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            color: #065F46;
        }
        .result-badge-pneumonia {
            background-color: #FEF2F2;
            border: 1px solid #FECACA;
            border-left: 5px solid #DC2626;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            color: #991B1B;
        }
        .status-pill-ok {
            display: inline-block;
            background: #E6F4EA;
            color: #137333;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .status-pill-warn {
            display: inline-block;
            background: #FEF3C7;
            color: #92400E;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .status-pill-info {
            display: inline-block;
            background: #E0F2FE;
            color: #0369A1;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_cached_model(model_path_str: str):
    """Load and cache the trained Keras model; return None if not trained yet."""
    model_path = Path(model_path_str)
    if not model_path.exists():
        return None
    return load_trained_model(model_path)


@st.cache_data
def get_dataset_status(data_dir_str: str, verify_images: bool = False) -> DatasetValidationResult:
    """Validate dataset directory and cache summary counts."""
    cfg = load_config()
    cfg.data_dir = Path(data_dir_str)
    return validate_dataset(cfg, verify_images=verify_images)


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_header() -> None:
    st.markdown(
        """
        <div class="xai-header">
            <p class="xai-title">XAI MEDSCAN</p>
            <p class="xai-subtitle">Explainable Medical Image Classification System &mdash; MobileNetV2 &amp; Grad-CAM</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        f'<div class="xai-disclaimer"><strong>Medical Disclaimer:</strong> {MEDICAL_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


def render_missing_model_box(config: ProjectConfig) -> None:
    st.warning(
        f"No trained model was found at `{config.model_path}`."
    )
    st.markdown(
        """
        <div class="xai-card">
            <h4>How to Train the Model</h4>
            <p>1. Ensure the Chest X-Ray dataset is extracted under <code>data/chest_xray/</code>.</p>
            <p>2. Run the training pipeline from PowerShell in the project root:</p>
            <pre><code>python -m src.train</code></pre>
            <p>3. Optionally evaluate on the held-out test set:</p>
            <pre><code>python -m src.evaluate</code></pre>
            <p>4. Refresh this application after training completes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_page(
    config: ProjectConfig,
    dataset_status: DatasetValidationResult,
    model_loaded: bool,
    metrics_data: Optional[Dict[str, Any]],
) -> None:
    render_disclaimer()

    col1, col2, col3 = st.columns(3)
    with col1:
        if dataset_status.is_valid:
            ds_badge = '<span class="status-pill-ok">Dataset: Ready</span>'
            ds_note = ""
        else:
            ds_badge = '<span class="status-pill-info">Dataset: Not Installed (Optional)</span>'
            ds_note = (
                '<p style="margin-top:0.5rem;margin-bottom:0;color:#475569;font-size:0.84rem;">'
                "The trained model supports image prediction without the original training dataset. "
                "The dataset is only required for local retraining and dataset integrity verification."
                "</p>"
            )
        st.markdown(
            f"""
            <div class="xai-card">
                <h4>Dataset Status</h4>
                <p>{ds_badge}</p>
                <p style="margin-bottom:0;color:#475569;font-size:0.9rem;">
                    Detected images: <strong>{dataset_status.total_images:,}</strong><br/>
                    Path: <code>data/chest_xray</code>
                </p>
                {ds_note}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        mdl_badge = (
            '<span class="status-pill-ok">Trained Model Loaded</span>'
            if model_loaded
            else '<span class="status-pill-warn">Not Trained Yet</span>'
        )
        st.markdown(
            f"""
            <div class="xai-card">
                <h4>Model Status</h4>
                <p>{mdl_badge}</p>
                <p style="margin-bottom:0;color:#475569;font-size:0.9rem;">
                    Architecture: <strong>MobileNetV2</strong><br/>
                    File: <code>{config.model_filename}</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        eval_badge = (
            '<span class="status-pill-ok">Evaluated</span>'
            if metrics_data is not None
            else '<span class="status-pill-warn">Pending Evaluation</span>'
        )
        acc_text = (
            f"{metrics_data['accuracy'] * 100:.2f}%"
            if metrics_data is not None
            else "N/A"
        )
        st.markdown(
            f"""
            <div class="xai-card">
                <h4>Evaluation Status</h4>
                <p>{eval_badge}</p>
                <p style="margin-bottom:0;color:#475569;font-size:0.9rem;">
                    Test Accuracy: <strong>{acc_text}</strong><br/>
                    Explainability: <strong>Grad-CAM</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown(
            """
            <div class="xai-card">
                <h4>Project Overview</h4>
                <p style="color:#334155;">
                    <strong>XAI MEDSCAN</strong> is an educational deep learning application designed to classify
                    pediatric chest X-ray images into <strong>NORMAL</strong> and <strong>PNEUMONIA</strong> using
                    transfer learning with <strong>MobileNetV2</strong>. To improve model transparency, every
                    prediction is paired with a <strong>Grad-CAM</strong> (Gradient-weighted Class Activation Mapping)
                    heatmap highlighting the spatial regions that influenced the network.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="xai-card">
                <h4>Model &amp; Classification Details</h4>
                <ul style="color:#334155;margin-bottom:0;">
                    <li><strong>Backbone:</strong> MobileNetV2 (ImageNet weights, <code>include_top=False</code>)</li>
                    <li><strong>Classification Head:</strong> GlobalAveragePooling2D &rarr; Dropout ({config.dropout_rate}) &rarr; Dense (1, Sigmoid)</li>
                    <li><strong>Input Resolution:</strong> {config.image_size[0]} &times; {config.image_size[1]} &times; 3 RGB</li>
                    <li><strong>Supported Classes:</strong> <code>NORMAL</code> (0) and <code>PNEUMONIA</code> (1, positive class)</li>
                    <li><strong>Explainability Method:</strong> TensorFlow <code>GradientTape</code> Grad-CAM</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="xai-card"><h4>Dataset Split Summary</h4>', unsafe_allow_html=True)
        if dataset_status.is_valid:
            st.dataframe(
                dataset_status.to_dataframe(),
                width="stretch",
                hide_index=True,
            )
            if dataset_status.warnings:
                for w in dataset_status.warnings:
                    st.caption(f"Note: {w}")
        else:
            st.info(
                "Dataset: Not Installed (Optional). "
                "The trained model supports image prediction without the original training dataset. "
                "The dataset is only required for local retraining and dataset integrity verification."
            )
            st.code(
                "data/\n"
                "└── chest_xray/\n"
                "    ├── train/\n"
                "    │   ├── NORMAL/\n"
                "    │   └── PNEUMONIA/\n"
                "    ├── val/\n"
                "    │   ├── NORMAL/\n"
                "    │   └── PNEUMONIA/\n"
                "    └── test/\n"
                "        ├── NORMAL/\n"
                "        └── PNEUMONIA/",
                language="text",
            )
            st.markdown(
                "Download source: [Kaggle Chest X-Ray Images (Pneumonia)]"
                "(https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)"
            )
        if st.button("Verify Dataset Integrity Now"):
            get_dataset_status.clear()
            verified = validate_dataset(config, verify_images=True)
            if verified.invalid_files:
                st.warning(f"Detected {len(verified.invalid_files)} unreadable image(s).")
            elif verified.is_valid:
                st.success(f"Dataset: Ready — verified {verified.total_images:,} valid image files.")
            else:
                st.info(
                    "Dataset: Not Installed (Optional). No local dataset was found at "
                    "`data/chest_xray/`. Prediction and Grad-CAM remain available via the trained model."
                )
        st.markdown("</div>", unsafe_allow_html=True)


def _run_and_store_prediction(
    model,
    uploaded_bytes: bytes,
    filename: str,
    config: ProjectConfig,
) -> Optional[PredictionResult]:
    try:
        stream = io.BytesIO(uploaded_bytes)
        result = predict_image(model, stream, config, include_gradcam=True)
        st.session_state["last_prediction"] = result
        st.session_state["last_filename"] = filename
        return result
    except ValueError as exc:
        st.error(f"Could not process the uploaded file: {exc}")
        return None


def render_prediction_page(config: ProjectConfig, model) -> None:
    st.subheader("Chest X-Ray Image Prediction")
    render_disclaimer()

    if model is None:
        render_missing_model_box(config)
        return

    st.info(
        "Intended Input: Upload a clear chest X-ray image (JPG, JPEG, or PNG). "
        "Arbitrary non-medical photographs are not meaningful chest radiographs and "
        "should not be interpreted clinically."
    )

    uploaded_file = st.file_uploader(
        "Upload a Chest X-Ray Image",
        type=["jpg", "jpeg", "png"],
        key="prediction_uploader",
    )

    result: Optional[PredictionResult] = None
    filename: Optional[str] = None

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        if len(raw_bytes) == 0:
            st.error("Uploaded file is empty. Please choose a valid image.")
            return
        result = _run_and_store_prediction(model, raw_bytes, uploaded_file.name, config)
        filename = uploaded_file.name
    elif "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        filename = st.session_state.get("last_filename", "Previous upload")

    if result is None:
        st.markdown(
            """
            <div class="xai-card">
                <h4>Awaiting Image Upload</h4>
                <p style="color:#475569;margin-bottom:0;">
                    Select a chest X-ray image above to view predicted class, confidence score,
                    class probabilities, and Grad-CAM visual explanation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    col_img, col_res = st.columns([1.0, 1.15])
    with col_img:
        st.markdown('<div class="xai-card"><h4>Uploaded Chest X-Ray</h4>', unsafe_allow_html=True)
        st.image(result.original_image, caption=filename, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        badge_cls = (
            "result-badge-pneumonia"
            if result.predicted_class == "PNEUMONIA"
            else "result-badge-normal"
        )
        st.markdown(
            f"""
            <div class="{badge_cls}">
                <div style="font-size:0.85rem;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;">
                    Predicted Category
                </div>
                <div style="font-size:1.8rem;font-weight:700;margin:0.15rem 0;">
                    {result.predicted_class}
                </div>
                <div style="font-size:1rem;">
                    Model Confidence: <strong>{result.confidence * 100:.2f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result.is_low_confidence:
            st.warning(
                f"Low-Confidence Prediction ({result.confidence * 100:.1f}% < "
                f"{config.low_confidence_threshold * 100:.0f}%): The model is uncertain "
                "between NORMAL and PNEUMONIA for this image."
            )

        st.markdown('<div class="xai-card" style="margin-top:1rem;"><h4>Class Probability Breakdown</h4>', unsafe_allow_html=True)
        st.write(f"**NORMAL:** {result.normal_probability * 100:.2f}%")
        st.progress(float(result.normal_probability))
        st.write(f"**PNEUMONIA (Positive Class):** {result.pneumonia_probability * 100:.2f}%")
        st.progress(float(result.pneumonia_probability))
        st.caption(
            f"Classification threshold = {config.classification_threshold:.2f} "
            "| Open the Explainability (Grad-CAM) tab to inspect spatial attribution."
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_explainability_page(config: ProjectConfig, model) -> None:
    st.subheader("Explainable AI: Grad-CAM Visualization")
    render_disclaimer()

    if model is None:
        render_missing_model_box(config)
        return

    uploaded_override = st.file_uploader(
        "Upload a Chest X-Ray Image (or use the image from Image Prediction)",
        type=["jpg", "jpeg", "png"],
        key="gradcam_uploader",
    )

    if uploaded_override is not None:
        raw_bytes = uploaded_override.getvalue()
        if len(raw_bytes) == 0:
            st.error("Uploaded file is empty.")
            return
        _run_and_store_prediction(model, raw_bytes, uploaded_override.name, config)

    result: Optional[PredictionResult] = st.session_state.get("last_prediction")
    filename = st.session_state.get("last_filename", "Uploaded Image")

    if result is None or result.gradcam is None:
        st.info("Upload a chest X-ray image above (or on the Image Prediction page) to generate Grad-CAM.")
        return

    gcam = result.gradcam
    st.markdown(
        f"""
        <div class="xai-card">
            <h4>Summary for <code>{filename}</code></h4>
            <p style="margin-bottom:0;color:#334155;">
                <strong>Predicted Class:</strong> {result.predicted_class} &nbsp;|&nbsp;
                <strong>Confidence:</strong> {result.confidence * 100:.2f}% &nbsp;|&nbsp;
                <strong>PNEUMONIA Score:</strong> {result.pneumonia_probability * 100:.2f}% &nbsp;|&nbsp;
                <strong>Target Conv Layer:</strong> <code>{gcam.target_layer_name}</code>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="xai-card"><h4>1. Original X-Ray</h4>', unsafe_allow_html=True)
        st.image(gcam.original_image, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="xai-card"><h4>2. Grad-CAM Heatmap</h4>', unsafe_allow_html=True)
        st.image(gcam.heatmap_image, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="xai-card"><h4>3. Heatmap Overlay</h4>', unsafe_allow_html=True)
        st.image(gcam.overlay_image, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.info(GRADCAM_DISCLAIMER)
    st.markdown(
        """
        <div class="xai-card">
            <h4>How Grad-CAM Works</h4>
            <p style="color:#334155;margin-bottom:0;">
                Grad-CAM computes the gradients of the <strong>PNEUMONIA</strong> output score with respect to
                the final convolutional feature maps of MobileNetV2 (<code>out_relu</code>) using
                <code>tf.GradientTape</code>. Those gradients are globally averaged to weight each feature channel,
                passed through ReLU to keep positive contributions, normalized to [0, 1], and upsampled onto the
                original radiograph. Warmer colors (red/yellow) show higher positive influence on the PNEUMONIA score.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_performance_page(
    config: ProjectConfig,
    metrics_data: Optional[Dict[str, Any]],
) -> None:
    st.subheader("Held-Out Test Set Performance")

    if metrics_data is None:
        st.warning("Model evaluation not yet completed.")
        st.markdown(
            """
            <div class="xai-card">
                <h4>Run Training &amp; Evaluation</h4>
                <p>No evaluation metrics file (<code>results/test_metrics.json</code>) was found.</p>
                <pre><code>python -m src.train
python -m src.evaluate</code></pre>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    caption_parts = []
    if metrics_data.get("num_test_samples"):
        caption_parts.append(f"Evaluated on {metrics_data['num_test_samples']} test images")
    caption_parts.append(
        f"Positive class: {metrics_data.get('positive_class', config.positive_class)}"
    )
    caption_parts.append(
        f"Threshold: {metrics_data.get('classification_threshold', config.classification_threshold):.2f}"
    )
    st.caption(" | ".join(caption_parts))

    def _fmt_pct(key: str) -> str:
        val = metrics_data.get(key)
        return f"{val * 100:.2f}%" if val is not None else "N/A"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", _fmt_pct("accuracy"))
    m2.metric("Precision", _fmt_pct("precision"))
    m3.metric("Recall", _fmt_pct("recall"))
    m4.metric("F1-Score", _fmt_pct("f1_score"))
    roc_val = metrics_data.get("roc_auc")
    m5.metric("ROC-AUC", f"{roc_val:.4f}" if roc_val is not None else "N/A")

    has_any_plot = any(
        p.exists()
        for p in (
            config.confusion_matrix_plot_path,
            config.roc_curve_plot_path,
            config.accuracy_plot_path,
            config.loss_plot_path,
        )
    )
    if not has_any_plot:
        st.info(
            "Evaluation plots (confusion matrix, ROC curve, and training curves) are not present in "
            "`results/`. Run `python -m src.evaluate` with the test dataset if you wish to generate them."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        if config.confusion_matrix_plot_path.exists():
            st.image(
                str(config.confusion_matrix_plot_path),
                caption="Confusion Matrix",
                width="stretch",
            )
    with col_b:
        if config.roc_curve_plot_path.exists():
            st.image(
                str(config.roc_curve_plot_path),
                caption="ROC Curve",
                width="stretch",
            )

    col_c, col_d = st.columns(2)
    with col_c:
        if config.accuracy_plot_path.exists():
            st.image(
                str(config.accuracy_plot_path),
                caption="Training & Validation Accuracy",
                width="stretch",
            )
    with col_d:
        if config.loss_plot_path.exists():
            st.image(
                str(config.loss_plot_path),
                caption="Training & Validation Loss",
                width="stretch",
            )

    if config.report_txt_path.exists():
        with st.expander("View Detailed Classification Report", expanded=False):
            st.code(config.report_txt_path.read_text(encoding="utf-8"), language="text")


def render_about_page(config: ProjectConfig) -> None:
    st.subheader("About the Project")
    render_disclaimer()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="xai-card">
                <h4>Project Objective</h4>
                <p style="color:#334155;">
                    Build an interpretable deep learning system for binary chest X-ray classification
                    (<strong>NORMAL</strong> vs <strong>PNEUMONIA</strong>) using transfer learning and
                    Grad-CAM visual explanations, tailored for academic demonstration and transparency in medical AI.
                </p>
                <h4>Technologies Used</h4>
                <ul style="color:#334155;margin-bottom:0;">
                    <li>Python, TensorFlow / Keras, MobileNetV2</li>
                    <li>NumPy, Pandas, Scikit-learn</li>
                    <li>Matplotlib, Pillow</li>
                    <li>Streamlit &amp; TensorFlow GradientTape (Grad-CAM)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="xai-card">
                <h4>Model Architecture &amp; Training Strategy</h4>
                <ul style="color:#334155;">
                    <li><strong>Base:</strong> MobileNetV2 (ImageNet pretrained, <code>include_top=False</code>)</li>
                    <li><strong>Head:</strong> GlobalAveragePooling2D &rarr; Dropout (0.35) &rarr; Dense (1, Sigmoid)</li>
                    <li><strong>Phase 1:</strong> Frozen backbone, head training (Adam lr=1e-4, 5 epochs)</li>
                    <li><strong>Phase 2:</strong> Unfreeze last 30 backbone layers (BatchNorm frozen), fine-tune (lr=1e-5, 3 epochs)</li>
                    <li><strong>Imbalance Handling:</strong> Balanced class weights + EarlyStopping + ReduceLROnPlateau</li>
                </ul>
                <h4>Dataset Source</h4>
                <p style="color:#334155;margin-bottom:0;">
                    <a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia" target="_blank">
                        Chest X-Ray Images (Pneumonia) &mdash; Kaggle
                    </a>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="xai-card">
            <h4>Known Limitations</h4>
            <ul style="color:#334155;margin-bottom:0;">
                <li>Trained on a pediatric chest X-ray cohort; generalization to adult or other clinical scanners is limited.</li>
                <li>Grad-CAM highlights coarse spatial regions (7&times;7 feature map upsampled to image size) and does not prove clinical causality.</li>
                <li>Cannot distinguish pneumonia subtypes (bacterial vs viral) or detect unrelated thoracic conditions.</li>
                <li>Not validated or approved for medical diagnosis.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sys_diag = config.assets_dir / "system_architecture.png"
    mdl_diag = config.assets_dir / "model_architecture.png"
    if sys_diag.exists() or mdl_diag.exists():
        d1, d2 = st.columns(2)
        with d1:
            if sys_diag.exists():
                st.image(str(sys_diag), caption="System Architecture", width="stretch")
        with d2:
            if mdl_diag.exists():
                st.image(str(mdl_diag), caption="MobileNetV2 Transfer Learning Architecture", width="stretch")


def main() -> None:
    inject_custom_css()
    config = load_config()
    dataset_status = get_dataset_status(str(config.data_dir), verify_images=False)
    model = get_cached_model(str(config.model_path))
    metrics_data = load_json_if_exists(config.metrics_path)

    with st.sidebar:
        st.markdown("### XAI MEDSCAN")
        st.caption("Explainable Medical Image AI")
        page = st.radio(
            "Navigation",
            [
                "Home / Dashboard",
                "Image Prediction",
                "Explainability (Grad-CAM)",
                "Model Performance",
                "About Project",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**System Status**")
        st.write(
            f"Dataset: {'Ready' if dataset_status.is_valid else 'Not Installed (Optional)'}"
        )
        st.write(f"Model: {'Loaded' if model is not None else 'Not Trained'}")
        st.write(f"Evaluation: {'Complete' if metrics_data is not None else 'Pending'}")
        st.divider()
        st.caption("Academic PBL prototype only. Not for clinical diagnosis.")

    render_header()

    if page == "Home / Dashboard":
        render_home_page(config, dataset_status, model is not None, metrics_data)
    elif page == "Image Prediction":
        render_prediction_page(config, model)
    elif page == "Explainability (Grad-CAM)":
        render_explainability_page(config, model)
    elif page == "Model Performance":
        render_performance_page(config, metrics_data)
    elif page == "About Project":
        render_about_page(config)


if __name__ == "__main__":
    main()

