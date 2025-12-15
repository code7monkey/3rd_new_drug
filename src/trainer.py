"""
Training routines for ChemBERTa experiments.

This module encapsulates the logic for performing k‑fold training using
HuggingFace's :class:`~transformers.Trainer` API.  Given a prepared
dataframe and corresponding cross‑validation splits, the training
function will iteratively train a model on each fold, evaluate it on
the held‑out data, save checkpoints and collect out‑of‑fold
predictions.  A manifest summarising the best checkpoint for each
fold is produced to facilitate downstream inference.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .model import load_model
from .utils import compute_regression_metrics, pIC50_to_IC50

__all__ = ["train_cv"]


def train_cv(
    df: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    tokenizer: AutoTokenizer,
    output_root: str,
    max_len: int = 256,
    learning_rate: float = 2e-5,
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    num_epochs: int = 10,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine_with_restarts",
    save_total_limit: int = 2,
    seed: int = 42,
    fp16: bool = True,
    dataloader_num_workers: int = 2,
) -> Dict[str, object]:
    """Run k‑fold cross‑validation and return training artefacts.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset containing at least the columns ``Smiles`` and
        ``pIC50``.  The indices of ``df`` must match those used in
        ``splits``.
    splits : list of (train_idx, val_idx)
        A list of index tuples defining the folds.  Typically produced
        by :func:`~chemberta_project.src.dataset.create_cv_splits`.
    model_name : str
        The HuggingFace model identifier to load for each fold.
    tokenizer : transformers.AutoTokenizer
        Tokeniser corresponding to ``model_name``.
    output_root : str
        Directory in which to store checkpoints, out‑of‑fold
        predictions and the manifest file.  One subdirectory per
        fold will be created.
    max_len : int, optional
        Maximum sequence length for tokenisation.  Longer SMILES
        strings will be truncated, shorter ones padded.
    learning_rate : float, optional
        Initial learning rate for the AdamW optimiser.
    train_batch_size : int, optional
        Per‑device training batch size.
    eval_batch_size : int, optional
        Per‑device evaluation batch size.
    num_epochs : int, optional
        Maximum number of training epochs.
    weight_decay : float, optional
        Weight decay to apply to all parameters except biases and
        LayerNorm weights.
    warmup_ratio : float, optional
        Fraction of total training steps used for warm‑up.
    lr_scheduler_type : str, optional
        Name of the learning rate scheduler to use.  See
        :class:`~transformers.TrainingArguments` for valid options.
    save_total_limit : int, optional
        Maximum number of checkpoints to keep on disk per fold.
    seed : int, optional
        Random seed for reproducibility.
    fp16 : bool, optional
        Whether to use half‑precision training when supported.
    dataloader_num_workers : int, optional
        Number of subprocesses to use for data loading.

    Returns
    -------
    dict
        A dictionary containing the following keys:

        ``oof_predictions`` (numpy.ndarray)
            Array of out‑of‑fold predictions aligned with ``df``.

        ``metrics`` (list of dict)
            Per‑fold evaluation metrics.

        ``manifest`` (dict)
            Information about each fold's best checkpoint,
            including the path, best epoch and best score.  This is
            saved to ``output_root/manifest.json``.
    """
    os.makedirs(output_root, exist_ok=True)
    n_folds = len(splits)

    # Prepare containers for out‑of‑fold predictions and metrics
    oof_pred_pic50 = np.zeros(len(df), dtype=np.float64)
    fold_metrics: List[Dict[str, float]] = []
    best_epochs: List[float | None] = []
    best_ckpts: List[str] = []
    best_scores: List[float] = []

    # Define function to tokenize and attach labels
    def tok_with_label(ex):
        out = tokenizer(
            ex["Smiles"],
            max_length=max_len,
            truncation=True,
            padding="max_length",
        )
        out["labels"] = float(ex["pIC50"])
        return out

    for fold_idx, (trn_idx, val_idx) in enumerate(splits, 1):
        print(f"\n===== FOLD {fold_idx}/{n_folds} =====")
        df_trn = df.iloc[trn_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        # Convert to HuggingFace datasets and tokenize
        ds_trn = Dataset.from_pandas(df_trn[["Smiles", "pIC50"]].copy()).map(tok_with_label)
        ds_val = Dataset.from_pandas(df_val[["Smiles", "pIC50"]].copy()).map(tok_with_label)
        for ds in (ds_trn, ds_val):
            ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Load a fresh model for each fold
        model = load_model(model_name, num_labels=1, problem_type="regression")

        # Training arguments specific to this fold
        out_dir = os.path.join(output_root, f"fold{fold_idx}")
        args = TrainingArguments(
            output_dir=out_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="score",
            greater_is_better=True,
            save_total_limit=save_total_limit,
            seed=seed,
            fp16=fp16,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=50,
            report_to="none",
            dataloader_num_workers=dataloader_num_workers,
        )

        # Define custom compute_metrics wrapper that uses our utils
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            metrics = compute_regression_metrics(preds, labels)
            return metrics

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_trn,
            eval_dataset=ds_val,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # Train and evaluate
        trainer.train()

        # Identify the best epoch and corresponding checkpoint
        best_metric = -float("inf")
        best_epoch: float | None = None
        for log in trainer.state.log_history:
            if "eval_score" in log and log["eval_score"] > best_metric:
                best_metric = float(log["eval_score"])
                best_epoch = log.get("epoch")
        # Fallback in case no evaluation was logged
        if best_epoch is None:
            last_eval = [l for l in trainer.state.log_history if "eval_score" in l]
            if last_eval:
                best_epoch = last_eval[-1].get("epoch")

        best_ckpt_path = trainer.state.best_model_checkpoint
        best_epochs.append(best_epoch)
        best_ckpts.append(best_ckpt_path)
        best_scores.append(best_metric)

        print(f"[FOLD {fold_idx}] ✅ Best eval_score={best_metric:.6f} @ epoch={best_epoch} | ckpt={best_ckpt_path}")

        # Predict on validation and fill OOF
        val_pred = trainer.predict(ds_val).predictions.reshape(-1)
        oof_pred_pic50[val_idx] = val_pred

        # Record fold metrics
        m = trainer.evaluate(ds_val)
        fold_metrics.append({k: float(v) for k, v in m.items()})
        print({k: round(float(v), 5) for k, v in m.items()})

    # Save out‑of‑fold predictions and fold metrics
    oof_df = pd.DataFrame({
        "Smiles": df["Smiles"],
        "pIC50_true": df["pIC50"].values.astype(np.float64),
        "pIC50_oof": oof_pred_pic50,
    })
    oof_path = os.path.join(output_root, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    metrics_path = os.path.join(output_root, "fold_metrics.csv")
    pd.DataFrame(fold_metrics).to_csv(metrics_path, index=False)

    # Build manifest for inference
    manifest = {
        "created_at": int(time.time()),
        "model_name": model_name,
        "max_len": max_len,
        "n_folds": n_folds,
        "folds": [],
    }
    for i, (ep, ck, sc) in enumerate(zip(best_epochs, best_ckpts, best_scores)):
        # Ensure epoch is serialisable (convert nan to None)
        epoch_val = None
        if ep is not None:
            try:
                # Convert floats very close to integers to int
                if abs(float(ep) - int(ep)) < 1e-9:
                    epoch_val = int(ep)
                else:
                    epoch_val = float(ep)
            except Exception:
                epoch_val = ep
        manifest["folds"].append(
            {
                "fold": i + 1,
                "best_epoch": epoch_val,
                "best_score": float(sc),
                "checkpoint": ck,
            }
        )
    manifest_path = os.path.join(output_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Per‑fold best checkpoints (for inference) ===")
    for f in manifest["folds"]:
        print(f"Fold {f['fold']}: epoch={f['best_epoch']}, score={f['best_score']:.6f}")
        print(f"           ckpt={f['checkpoint']}")

    return {
        "oof_predictions": oof_pred_pic50,
        "metrics": fold_metrics,
        "manifest": manifest,
    }