"""
Dataset loading and preparation utilities for ChemBERTa experiments.

This module is responsible for ingesting raw CSV files containing
SMILES strings and activity data, cleaning and canonicalising the
records, converting IC50 values to the pIC50 scale and constructing
stratified group splits based on molecular scaffolds.  The stratified
splitting strategy follows the approach outlined in the provided
notebooks: the continuous pIC50 values are binned into quantiles to
ensure even representation across folds, while scaffolds serve as
group identifiers to avoid leakage of structurally similar compounds
across folds.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Tuple

from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import Chem

from .utils import canonicalize_smiles, IC50_to_pIC50

__all__ = [
    "load_and_process_data",
    "create_cv_splits",
]


def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load a CSV file and return a cleaned dataframe of SMILES and pIC50.

    The input CSV is expected to contain at least two columns: a
    column named ``Smiles`` (case sensitive) holding the molecular
    SMILES strings, and either ``pIC50`` or ``ic50_nM`` holding the
    activity data.  If ``pIC50`` is absent, the IC50 values are
    automatically converted to pIC50.  Rows with missing or invalid
    data are dropped.  Canonicalisation is applied to the SMILES
    strings and duplicates are aggregated by taking the median pIC50.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file to load.

    Returns
    -------
    pandas.DataFrame
        A dataframe with two columns: ``Smiles`` and ``pIC50``.  The
        dataframe is sorted by SMILES and has no missing values.
    """
    df = pd.read_csv(csv_path)
    if "Smiles" not in df.columns:
        raise ValueError(f"Expected column 'Smiles' in {csv_path}")

    # Determine whether pIC50 needs to be computed
    if "pIC50" in df.columns:
        df["pIC50"] = pd.to_numeric(df["pIC50"], errors="coerce")
    elif "ic50_nM" in df.columns:
        df["ic50_nM"] = pd.to_numeric(df["ic50_nM"], errors="coerce")
        df["pIC50"] = IC50_to_pIC50(df["ic50_nM"])
    else:
        raise ValueError(
            f"No activity column found in {csv_path}. Expected 'pIC50' or 'ic50_nM'."
        )

    # Filter out invalid or out‑of‑range values (matching the notebook)
    df = df.dropna(subset=["Smiles", "pIC50"]).copy()
    df["Smiles"] = df["Smiles"].map(canonicalize_smiles)
    df = df.dropna(subset=["Smiles"]).copy()
    # Only keep pIC50 values between 6 and 10 inclusive
    df = df[(df["pIC50"] >= 6.0) & (df["pIC50"] <= 10.0)]

    # Aggregate duplicate SMILES by their median pIC50
    df = (
        df.groupby("Smiles", as_index=False)["pIC50"]
        .median()
        .reset_index(drop=True)
    )
    return df


def _smiles_to_scaffold(smi: str) -> str:
    """Compute the Bemis–Murcko scaffold for a SMILES string.

    If RDKit fails to parse the SMILES, the sentinel string ``"NONE"``
    is returned instead.  Scaffolds are used as group identifiers
    during cross‑validation to prevent structurally similar molecules
    from being split across different folds.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "NONE"
    return MurckoScaffoldSmiles(mol=mol, includeChirality=True)


def create_cv_splits(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> List[Tuple[pd.Index, pd.Index]]:
    """Create stratified group k‑fold splits based on pIC50 and scaffolds.

    The dataframe must contain the columns ``pIC50`` and ``Smiles``.
    This function adds two temporary columns:

    * ``bin`` – quantile labels of pIC50 for stratification.
    * ``scaffold`` – scaffold strings for grouping.

    These columns are removed from the returned dataframe; they are
    only used to compute the splits.  The returned splits are
    suitable for passing to scikit‑learn or HuggingFace training code.

    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed dataframe returned by
        :func:`load_and_process_data`.
    n_folds : int
        Number of folds to create.
    seed : int
        Random seed for reproducible splitting.

    Returns
    -------
    list of tuples
        A list of ``(train_idx, val_idx)`` tuples, each containing
        integer indices into ``df``.
    """
    if "pIC50" not in df.columns:
        raise ValueError("Column 'pIC50' missing from dataframe")
    if "Smiles" not in df.columns:
        raise ValueError("Column 'Smiles' missing from dataframe")

    tmp_df = df.copy()
    # Bin pIC50 into quantiles for stratification; duplicates are dropped
    q = min(10, len(tmp_df))
    tmp_df["bin"] = pd.qcut(tmp_df["pIC50"], q=q, labels=False, duplicates="drop")
    # Compute scaffolds
    tmp_df["scaffold"] = tmp_df["Smiles"].map(_smiles_to_scaffold)

    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(cv.split(tmp_df, y=tmp_df["bin"], groups=tmp_df["scaffold"]))
    return splits