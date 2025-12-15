"""
chemberta_project.src
======================

This package contains the core modules used to build, train and
run inference with the ChemBERTa model.  The code has been
modularised so that it can be imported from a single namespace
(`chemberta_project.src`), making it easy to reuse components in
custom scripts or notebooks.  Modules include:

* ``utils`` – common utility functions such as random seed
  initialisation and SMILES canonicalisation.
* ``dataset`` – helpers for loading and preparing datasets, as well
  as creating stratified scaffold splits.
* ``model`` – a thin wrapper around HuggingFace to load a
  regression‐style sequence classifier.
* ``trainer`` – functions to orchestrate multi‑fold training and
  generate output files such as manifests and out‑of‑fold
  predictions.
* ``losses`` – a placeholder for any custom loss functions you may
  wish to implement in the future.

To use this package in your own code, simply add the root
directory to ``PYTHONPATH`` or install it as a package.  See
``train.py`` and ``inference.py`` at the project root for usage
examples.
"""

__all__ = [
    "utils",
    "dataset",
    "model",
    "trainer",
    "losses",
]