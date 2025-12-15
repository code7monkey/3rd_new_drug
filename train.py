"""
Entry point for training ChemBERTa models.

This script reads experiment parameters from a YAML configuration file
and runs multi‑fold training accordingly.  It orchestrates data
loading, splitting, tokenisation and training via the helper
functions defined in ``chemberta_project.src``.  Upon completion, it
produces a set of checkpoints, a manifest describing the best
checkpoint per fold, and CSV files summarising out‑of‑fold
predictions and fold metrics.

Example
-------

Run training using the default configuration in ``configs/train.yaml``::

    python train.py --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import os
import yaml

from transformers import AutoTokenizer

from src.utils import set_seed
from src.dataset import load_and_process_data, create_cv_splits
from src.trainer import train_cv


def main(config_path: str) -> None:
    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Set up reproducibility
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # Data loading and splitting
    dataset_path = cfg["dataset_path"]
    df = load_and_process_data(dataset_path)
    n_folds = int(cfg.get("n_folds", 5))
    splits = create_cv_splits(df, n_folds=n_folds, seed=seed)

    # Model and tokenizer
    model_name = cfg.get("model_name", "DeepChem/ChemBERTa-77M-MTR")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Output directory
    output_root = cfg.get("output_root", "./outputs")
    os.makedirs(output_root, exist_ok=True)

    # Training hyperparameters
    max_len = int(cfg.get("max_len", 256))
    learning_rate = float(cfg.get("learning_rate", 2e-5))
    train_batch_size = int(cfg.get("train_batch_size", 32))
    eval_batch_size = int(cfg.get("eval_batch_size", 64))
    num_epochs = int(cfg.get("num_epochs", 10))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.1))
    lr_scheduler_type = cfg.get("lr_scheduler_type", "cosine_with_restarts")
    save_total_limit = int(cfg.get("save_total_limit", 2))

    # Start cross‑validation training
    train_cv(
        df=df,
        splits=splits,
        model_name=model_name,
        tokenizer=tokenizer,
        output_root=output_root,
        max_len=max_len,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        save_total_limit=save_total_limit,
        seed=seed,
        fp16=bool(cfg.get("fp16", True)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChemBERTa model with k‑fold cross‑validation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    main(args.config)