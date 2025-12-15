"""
Inference script for ChemBERTa models.

This script reads a manifest produced during training, loads the
best checkpoint from each fold, computes predictions on a test
dataset and averages them to produce a final submission file.  The
configuration file used here (default: ``configs/submit.yaml``)
specifies the paths to the manifest, the test CSV and the output
submission filename.

Example
-------

Run inference with the default configuration::

    python inference.py --config configs/submit.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import yaml
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments

from src.utils import canonicalize_smiles, pIC50_to_IC50

import torch  # for fp16 check


def main(config_path: str) -> None:
    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    manifest_path = cfg.get("manifest_path")
    if manifest_path is None:
        raise ValueError("'manifest_path' must be specified in the submit configuration")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    model_name = manifest["model_name"]
    max_len = int(manifest.get("max_len", 256))
    fold_entries = manifest.get("folds", [])
    if not fold_entries:
        raise ValueError("No fold information found in manifest")

    # Test data
    test_csv_path = cfg.get("test_csv_path")
    if test_csv_path is None:
        raise ValueError("'test_csv_path' must be specified in the submit configuration")
    test_df = pd.read_csv(test_csv_path)
    if "Smiles" not in test_df.columns:
        raise ValueError("Test CSV must contain a 'Smiles' column")
    if "ID" not in test_df.columns:
        # If no ID column is present, create a default one
        test_df["ID"] = np.arange(len(test_df))
    test_df["Smiles"] = test_df["Smiles"].map(canonicalize_smiles)

    # Prepare dataset for tokenisation
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok_test_only(ex):
        return tokenizer(
            ex["Smiles"],
            max_length=max_len,
            truncation=True,
            padding="max_length",
        )

    ds_test = Dataset.from_pandas(test_df[["Smiles"]].copy())
    ds_test = ds_test.map(tok_test_only)
    ds_test.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Predict with each fold's checkpoint
    all_fold_preds: list[np.ndarray] = []
    for f in fold_entries:
        ckpt_dir = f.get("checkpoint")
        if not ckpt_dir:
            raise ValueError("Checkpoint path missing in manifest fold entry")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

        config = AutoConfig.from_pretrained(ckpt_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir, config=config)
        # Build a minimal TrainingArguments object for prediction
        tmp_out = os.path.join(os.path.dirname(manifest_path), f"infer_tmp_fold{f['fold']}")
        args = TrainingArguments(
            output_dir=tmp_out,
            per_device_eval_batch_size=int(cfg.get("eval_batch_size", 64)),
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
            report_to="none",
        )
        trainer = Trainer(model=model, args=args, tokenizer=tokenizer)
        preds = trainer.predict(ds_test).predictions.reshape(-1)
        all_fold_preds.append(preds)

    # Soft ensemble: average predictions
    test_pic50_mean = np.mean(np.stack(all_fold_preds, axis=0), axis=0)
    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "ASK1_IC50_nM": pIC50_to_IC50(test_pic50_mean),
    })

    submission_name = cfg.get("submission_name", manifest.get("submission_name", "submission.csv"))
    sub_path = os.path.join(os.path.dirname(manifest_path), submission_name)
    submission.to_csv(sub_path, index=False)
    print(f"Saved submission: {sub_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a trained ChemBERTa model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/submit.yaml",
        help="Path to the YAML submit configuration file",
    )
    args = parser.parse_args()
    main(args.config)