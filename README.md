# 3rd_new_drug

**Jump AI (Python) 2025 – 3rd AI Drug Discovery Challenge**  
**🥈 2nd Place / 502 Teams – ChemBERTa pIC50 Prediction**

---

This repository contains a **ChemBERTa-based regression pipeline** for predicting **pIC50 values from SMILES**, developed for the **ASK1 (MAP3K5) target**.

The project is designed with **clear separation between training and inference**, and all experiments are managed through **YAML configuration files** for reproducibility and flexibility.

---

## 🎯 Project Goals

- **pIC50 regression from SMILES inputs**
- **Scaffold-based Stratified Group K-Fold cross-validation**
- **Stable training using HuggingFace Trainer**
- **Soft-ensemble inference using best checkpoints from each fold**

---

## 📁 Project Structure

    chemberta_project/
    ├── src/                    # Core logic (importable modules)
    │   ├── __init__.py
    │   ├── model.py            # Model definition & loading
    │   ├── dataset.py          # Data preprocessing & CV split
    │   ├── trainer.py          # Cross-validation training loop
    │   ├── losses.py           # (Optional) custom loss functions
    │   └── utils.py            # Shared utilities
    │
    ├── train.py                # Training entry point
    ├── inference.py            # Inference & submission generation
    │
    ├── configs/                # Experiment configs (YAML-based)
    │   ├── train.yaml
    │   └── submit.yaml
    │
    ├── assets/                 # Model weights / tokenizer (gitignored)
    │   ├── model.pt
    │   └── tokenizer/
    │
    ├── requirements.txt        # Fixed environment dependencies
    ├── .gitignore
    ├── .gitattributes
    └── README.md

---

## 🛠 Environment Setup

Python **3.9+** is recommended.

    pip install -r requirements.txt

---

## 📊 Dataset Format

The dataset consists of **two columns only**: `ID` and `Smiles`.

    ID,Smiles
    TEST_000,CCO...
    TEST_001,CCN...

---

## 🚀 Training

### Configure Training Settings

Edit `configs/train.yaml` to control:

- Pretrained model (e.g. `DeepChem/ChemBERTa-77M-MTR`)
- Batch size, epochs, learning rate
- Number of folds
- Output directories

### Run Training

    python train.py --config configs/train.yaml

After training, the following artifacts are generated:

- Best checkpoint per fold
- Out-of-fold predictions (`oof_*.csv`)
- `manifest.json` (used for inference)

---

## 📦 Inference & Submission

    python inference.py --config configs/submit.yaml

Inference pipeline:

- Loads best checkpoint from each fold
- Averages fold-wise pIC50 predictions (soft ensemble)
- Converts **pIC50 → IC50 (nM)**
- Saves final submission file

---

## 🧠 Model Details

- **Backbone**: ChemBERTa-77M-MTR  
- **Task**: Regression (pIC50)  
- **Loss Function**: Mean Squared Error (MSE)  
- **Cross-Validation Strategy**:
  - Murcko scaffold-based grouping  
  - Stratification using binned pIC50 values  

---

## 📌 Notes

- `assets/`, `data/`, and `ckpt/` directories are excluded via `.gitignore`
- Git LFS is recommended for large model files
- `losses.py` is prepared for future experiments with custom loss functions
