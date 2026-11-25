# Explainable Deep Learning-Based Medical Image Classification Using MobileNetV2 and Grad-CAM

**Application Name:** **XAI MEDSCAN** (*Explainable Medical Image Classification System*)

> **Important Medical Disclaimer:** This project is an academic B.Tech Machine Learning Project-Based Learning (PBL) prototype built strictly for educational demonstration. It is **NOT** a medical diagnostic system and must **NOT** be used for clinical decisions.

---

## 1. Project Introduction

Deep learning models can classify chest radiographs with high accuracy, but standard convolutional neural networks act as "black boxes." Clinicians and students often need to see *which regions* of an X-ray drove a prediction.

**XAI MEDSCAN** combines **MobileNetV2** transfer learning with **Grad-CAM (Gradient-weighted Class Activation Mapping)** inside a clean **Streamlit** web dashboard to classify chest X-rays into **NORMAL** or **PNEUMONIA** while visualizing spatial feature importance.

---

## 2. Problem Statement & Objectives

- **Problem:** Automated medical image classifiers often lack visual interpretability, making them difficult to inspect, trust, and explain in academic or clinical settings.
- **Objectives:**
  1. Validate and preprocess the public Chest X-Ray Images (Pneumonia) dataset consistently.
  2. Train a lightweight **MobileNetV2** transfer-learning model suitable for a standard student laptop.
  3. Handle class imbalance using balanced class weights and prevent overfitting using Dropout, data augmentation, `EarlyStopping`, and `ReduceLROnPlateau`.
  4. Evaluate the held-out test set using Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, and ROC Curve (with `PNEUMONIA` as the positive class).
  5. Generate **Grad-CAM** heatmaps and overlays using TensorFlow `GradientTape`.
  6. Deliver a clean, responsive **Streamlit** interface (**XAI MEDSCAN**) with graceful error handling.

---

## 3. Technologies Used

- **Language:** Python 3.11
- **Deep Learning:** TensorFlow / Keras, MobileNetV2 (ImageNet pretrained)
- **Explainable AI (XAI):** Grad-CAM via `tf.GradientTape`
- **Data & Metrics:** NumPy, Pandas, Scikit-learn
- **Visualization & Imaging:** Matplotlib, Pillow
- **Web Interface:** Streamlit

---

## 4. Project Architecture

```text
explainable_medical _ai/
├── .streamlit/
│   └── config.toml                 # White & soft-blue healthcare theme
├── app/
│   └── streamlit_app.py            # XAI MEDSCAN Streamlit dashboard
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized settings & config.json loader
│   ├── data_utils.py               # Dataset validation, preprocessing & augmentation
│   ├── model.py                    # MobileNetV2 builder, freezing & fine-tuning
│   ├── train.py                    # Two-stage training pipeline
│   ├── evaluate.py                 # Held-out test evaluation & plots
│   ├── gradcam.py                  # GradientTape Grad-CAM implementation
│   └── predict.py                  # Single-image prediction & CLI
├── data/
│   └── README.md                   # Dataset setup instructions
├── models/                         # Saved .keras models (ignored in git)
├── results/                        # Metrics JSON, reports & plots
├── assets/
│   ├── generate_diagrams.py
│   ├── system_architecture.png
│   └── model_architecture.png
├── tests/
│   └── verify_project.py           # Automated verification suite
├── config.json                     # Editable training/inference hyperparameters
├── requirements.txt
├── README.md
├── .gitignore
└── run.bat
```

### Key Design Decisions (Viva Explanation Guide)

1. **Why MobileNetV2?** It uses depthwise separable convolutions and inverted residual blocks, making it fast and memory-efficient enough to train and run on a student laptop CPU without sacrificing accuracy.
2. **Why two-stage training?** First, we freeze the ImageNet backbone and train only the new classification head (`lr=1e-4`) so random head gradients do not destroy pretrained filters. Second, we unfreeze only the last 30 backbone layers (`lr=1e-5`) to adapt high-level features to chest X-rays.
3. **Why keep Batch Normalization frozen during fine-tuning?** Small batch sizes (e.g., 16) destabilize batch statistics; keeping `BatchNormalization` layers frozen preserves stable ImageNet feature scaling.
4. **Why class weights?** Chest X-ray datasets contain more `PNEUMONIA` than `NORMAL` training images; balanced class weights penalize minority-class mistakes proportionally.
5. **Why Grad-CAM?** Grad-CAM uses gradients of the `PNEUMONIA` score with respect to the last convolutional feature map (`out_relu`) to highlight influential regions without modifying the model architecture.

---

## 5. Dataset Download Instructions

1. Download the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:
   <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>
2. Extract the archive so the directory matches:

```text
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

*(If the zip extracts as `chest_xray/chest_xray/`, move the inner `train/`, `val/`, and `test/` folders directly inside `data/chest_xray/`.)*

---

## 6. Installation & Virtual Environment Setup (Windows PowerShell)

```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 7. Running the Project (Windows PowerShell)

### Train the Model
```powershell
python -m src.train
```
*(Optional overrides: `python -m src.train --epochs 5 --fine-tune-epochs 3 --batch-size 16`, or edit `config.json`.)*

### Evaluate on Held-Out Test Set
```powershell
python -m src.evaluate
```

### Run CLI Single-Image Prediction + Grad-CAM
```powershell
python -m src.predict --image "data\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg" --save-overlay "results\sample_overlay.png"
```

### Launch the Streamlit Web App
```powershell
streamlit run app/streamlit_app.py
```
Or simply double-click `run.bat` (or run `.\run.bat` in PowerShell).

---

## 8. Expected Output Files

- **Model:** `models/mobilenetv2_pneumonia_best.keras`
- **Training History & Curves:**
  - `results/training_history.json`
  - `results/accuracy_curve.png`
  - `results/loss_curve.png`
- **Test Evaluation Outputs:**
  - `results/test_metrics.json`
  - `results/classification_report.txt` & `results/classification_report.json`
  - `results/confusion_matrix.png`
  - `results/roc_curve.png`

---

## 9. Verification Suite

Run the built-in verification suite to validate project structure, imports, dataset validation logic, model construction, preprocessing, prediction, and Grad-CAM:

```powershell
python tests/verify_project.py
```

---

## 10. Troubleshooting

- **`Dataset is incomplete or missing`:** Confirm `train`, `val`, and `test` folders (each with `NORMAL` and `PNEUMONIA`) sit directly inside `data/chest_xray/`.
- **`Trained model not found`:** Run `python -m src.train` before running `src.evaluate`, `src.predict`, or uploading images in Streamlit.
- **PowerShell script execution policy error during `Activate.ps1`:** Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or invoke `.venv\Scripts\python.exe` directly.
- **Slow training on CPU:** Reduce `batch_size` or epochs in `config.json` (default `5 + 3` epochs is already sized for CPU laptops).

---

## 11. Limitations & Medical Disclaimer

- Trained only on pediatric chest X-rays from the Kaggle Pneumonia dataset; performance may drop on external hospitals or adult radiographs.
- Grad-CAM produces coarse spatial heatmaps and serves only as an approximate interpretability aid; it does **not** establish medical causality.
- **Academic Disclaimer:** This software is strictly for educational demonstration and must **never** be used for medical diagnosis or treatment.

