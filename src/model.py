"""
Model loading utilities.

This module provides helper functions to load a HuggingFace
transformer model configured for regression tasks.  By abstracting
model construction into its own module you can easily swap out
different architectures or modify the configuration without
changing your training scripts.
"""

from __future__ import annotations

from typing import Optional

from transformers import AutoConfig, AutoModelForSequenceClassification

__all__ = ["load_model"]


def load_model(
    model_name: str,
    num_labels: int = 1,
    problem_type: str = "regression",
    checkpoint_path: Optional[str] = None,
) -> AutoModelForSequenceClassification:
    """Load a sequence classification model configured for regression.

    Parameters
    ----------
    model_name : str
        The name of a pre‑trained model on HuggingFace Hub.  For
        ChemBERTa experiments this is typically
        ``"DeepChem/ChemBERTa-77M-MTR"``, but any encoder model that
        supports sequence classification can be used.
    num_labels : int, optional
        Number of output labels.  For regression this should be 1.
    problem_type : str, optional
        The problem type passed to the configuration.  Supported
        values include ``"regression"`` and ``"single_label_classification"``.
    checkpoint_path : str, optional
        If provided, the model weights will be loaded from the
        specified checkpoint directory instead of the hub.  This can
        be useful when resuming training or performing inference on
        previously saved checkpoints.

    Returns
    -------
    transformers.AutoModelForSequenceClassification
        A HuggingFace model ready for training or inference.
    """
    config = AutoConfig.from_pretrained(
        checkpoint_path or model_name,
        num_labels=num_labels,
        problem_type=problem_type,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path or model_name,
        config=config,
    )
    return model