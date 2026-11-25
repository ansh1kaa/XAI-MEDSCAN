"""Centralized configuration for training, evaluation, Grad-CAM, and inference.

Users can customize training hyperparameters either by editing `config.json` in
the project root (created automatically on demand or edited directly) or by
modifying the defaults below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ProjectConfig:
    """Configuration container for the XAI MedScan pipeline."""

    # Reproducibility
    seed: int = 42

    # Paths (relative to project root unless absolute)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "chest_xray")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    assets_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "assets")
    model_filename: str = "mobilenetv2_pneumonia_best.keras"

    # Classes: NORMAL = 0, PNEUMONIA = 1 (positive class)
    class_names: Tuple[str, str] = ("NORMAL", "PNEUMONIA")
    positive_class: str = "PNEUMONIA"
    positive_class_index: int = 1
    supported_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

    # Input & preprocessing
    image_size: Tuple[int, int] = (224, 224)
    channels: int = 3

    # Training hyperparameters (laptop-friendly defaults)
    batch_size: int = 16
    initial_epochs: int = 5
    fine_tune_epochs: int = 3
    initial_learning_rate: float = 1e-4
    fine_tune_learning_rate: float = 1e-5
    fine_tune_last_n_layers: int = 30
    dropout_rate: float = 0.35

    # Callbacks
    early_stopping_patience: int = 3
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7

    # Augmentation (training split only)
    rotation_factor: float = 0.04   # ~14 degrees
    zoom_factor: float = 0.10       # +/- 10% zoom

    # Inference & Grad-CAM
    classification_threshold: float = 0.50
    low_confidence_threshold: float = 0.70
    gradcam_alpha: float = 0.40
    gradcam_colormap: str = "jet"
    preferred_gradcam_layer: str = "out_relu"

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename

    @property
    def history_path(self) -> Path:
        return self.results_dir / "training_history.json"

    @property
    def metrics_path(self) -> Path:
        return self.results_dir / "test_metrics.json"

    @property
    def report_txt_path(self) -> Path:
        return self.results_dir / "classification_report.txt"

    @property
    def report_json_path(self) -> Path:
        return self.results_dir / "classification_report.json"

    @property
    def accuracy_plot_path(self) -> Path:
        return self.results_dir / "accuracy_curve.png"

    @property
    def loss_plot_path(self) -> Path:
        return self.results_dir / "loss_curve.png"

    @property
    def confusion_matrix_plot_path(self) -> Path:
        return self.results_dir / "confusion_matrix.png"

    @property
    def roc_curve_plot_path(self) -> Path:
        return self.results_dir / "roc_curve.png"

    def ensure_directories(self) -> None:
        """Ensure output directories exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, object]:
        """Serialize configuration to JSON-friendly dictionary."""
        raw = asdict(self)
        for key in ("data_dir", "models_dir", "results_dir", "assets_dir"):
            raw[key] = str(raw[key])
        raw["class_names"] = list(self.class_names)
        raw["supported_extensions"] = list(self.supported_extensions)
        raw["image_size"] = list(self.image_size)
        return raw


def load_config(config_file: Path | str | None = None) -> ProjectConfig:
    """Load default config and apply optional overrides from `config.json`."""
    cfg = ProjectConfig()
    target = Path(config_file) if config_file else (PROJECT_ROOT / "config.json")
    if not target.exists():
        return cfg

    with target.open("r", encoding="utf-8") as f:
        overrides = json.load(f)

    path_fields = {"data_dir", "models_dir", "results_dir", "assets_dir"}
    tuple_fields = {"class_names", "supported_extensions", "image_size"}

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            continue
        if key in path_fields:
            p = Path(value)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            setattr(cfg, key, p)
        elif key in tuple_fields:
            setattr(cfg, key, tuple(value))
        else:
            setattr(cfg, key, value)

    return cfg

