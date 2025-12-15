"""
Utility functions for data processing, reproducibility and metric
calculation.

The functions defined here are intentionally generic so they can be
reused across training and inference scripts.  For example,
``set_seed`` ensures deterministic behaviour across Python, NumPy
and PyTorch; ``canonicalize_smiles`` converts an input SMILES string
to a canonical form using RDKit; and conversion functions between
IC50 and pIC50 allow for consistent units.
"""

from __future__ import annotations

import os
import random
from typing import Optional, Sequence

import numpy as np
import torch
from rdkit import Chem

__all__ = [
    "set_seed",
    "canonicalize_smiles",
    "IC50_to_pIC50",
    "pIC50_to_IC50",
    "compute_regression_metrics",
]


def set_seed(seed: int) -> None:
    """Set the random seed for Python, NumPy and PyTorch.

    Parameters
    ----------
    seed : int
        A non‑negative integer to initialise the random number
        generators.  Setting the seed ensures that experiments are
        reproducible, which is particularly important when using
        cross‑validation or stochastic optimisation.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonicalize_smiles(smi: Optional[str]) -> Optional[str]:
    """Convert a SMILES string to its canonical form.

    When working with chemical structures, it is often desirable to
    canonicalise SMILES strings so that the same molecule is always
    represented by the same text.  RDKit performs this
    canonicalisation for you.  Invalid SMILES or missing values
    return ``None``.

    Parameters
    ----------
    smi : Optional[str]
        The input SMILES string.

    Returns
    -------
    Optional[str]
        The canonicalised SMILES, or ``None`` if the input could
        not be parsed.
    """
    if smi is None:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smi))
    except Exception:
        return None
    return (
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if mol is not None
        else None
    )


def IC50_to_pIC50(ic50_nM: float | Sequence[float]) -> float | np.ndarray:
    """Convert IC50 values in nanomolar units to pIC50.

    The pIC50 scale is defined as ``9 – log10(IC50_nM)``.  To avoid
    taking the logarithm of zero or negative values, inputs are
    clipped to a minimum of ``1e‑10``.

    Parameters
    ----------
    ic50_nM : float or Sequence[float]
        One or more IC50 values in nanomolar units.

    Returns
    -------
    float or numpy.ndarray
        The corresponding pIC50 values.
    """
    # Convert to numpy array for vectorised operations
    arr = np.asarray(ic50_nM, dtype=np.float64)
    arr = np.clip(arr, 1e-10, None)
    pic50 = 9.0 - np.log10(arr)
    return pic50 if isinstance(ic50_nM, Sequence) else float(pic50)


def pIC50_to_IC50(pIC50: float | Sequence[float]) -> float | np.ndarray:
    """Convert pIC50 values back to IC50 in nanomolar units.

    The reverse transformation of :func:`IC50_to_pIC50` is given by
    ``IC50_nM = 10**(9 – pIC50)``.

    Parameters
    ----------
    pIC50 : float or Sequence[float]
        One or more pIC50 values.

    Returns
    -------
    float or numpy.ndarray
        The corresponding IC50 values in nanomolar units.
    """
    arr = np.asarray(pIC50, dtype=np.float64)
    ic50 = np.power(10.0, 9.0 - arr)
    return ic50 if isinstance(pIC50, Sequence) else float(ic50)


def compute_regression_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute evaluation metrics for a regression task on pIC50.

    The metrics computed here mirror those used in the original
    notebook: an RMSE normalised by the range of IC50 values (A), the
    squared Pearson correlation coefficient (B) and their weighted
    combination (score).  Additional metrics such as RMSE, MAE and
    R² are also returned for convenience.

    Parameters
    ----------
    preds : numpy.ndarray
        Predicted pIC50 values, shape ``(n_samples,)``.
    labels : numpy.ndarray
        True pIC50 values, shape ``(n_samples,)``.

    Returns
    -------
    dict[str, float]
        A dictionary containing the computed metrics.
    """
    # ensure shapes
    preds = preds.reshape(-1).astype(np.float64)
    labels = labels.reshape(-1).astype(np.float64)

    # Convert pIC50 to IC50 for normalised RMSE
    y_true_ic50 = pIC50_to_IC50(labels)
    y_pred_ic50 = pIC50_to_IC50(preds)
    rmse_ic50 = float(np.sqrt(np.mean((y_true_ic50 - y_pred_ic50) ** 2)))
    value_range = float(np.max(y_true_ic50) - np.min(y_true_ic50))
    A = float(rmse_ic50 / (value_range if value_range > 1e-12 else 1.0))

    # Squared Pearson correlation coefficient
    x = labels - labels.mean()
    y = preds - preds.mean()
    denom = float(np.sqrt((x**2).sum()) * np.sqrt((y**2).sum()))
    r = float((x * y).sum() / denom) if denom > 0 else 0.0
    B = float(r * r)

    # Standard regression metrics
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    mae = float(np.mean(np.abs(preds - labels)))
    ss_res = float(np.sum((labels - preds) ** 2))
    ss_tot = float(np.sum((labels - labels.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Weighted combination (used as the optimisation objective in the notebook)
    score = 0.4 * (1.0 - min(A, 1.0)) + 0.6 * B

    return {
        "score": score,
        "A_nrmse": A,
        "B_r2": B,
        "rmse_ic50": rmse_ic50,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }