# Source Code

This directory contains reusable modules for the image classification project.

## Files

| File | Description |
|---|---|
| `config.py` | Stores dataset, model, training, and ensemble configuration. |
| `preprocessing.py` | Contains utilities for loading metadata and preprocessing images. |
| `dataset.py` | Defines the PyTorch dataset and data loading utilities. |
| `dit_model.py` | Provides functions for building and running the DiT/BEiT model. |
| `convnext_model.py` | Provides functions for building and running the ConvNeXt Tiny model. |
| `ensemble.py` | Combines model probabilities and searches for the best ensemble weight. |
| `evaluation.py` | Calculates classification metrics and confusion matrix. |

## Models

The project uses two image classification models:

- DiT/BEiT (`microsoft/dit-base`)
- ConvNeXt Tiny (`convnext_tiny.fb_in22k_ft_in1k`)

Their prediction probabilities are combined using weighted ensembling.

## Notes

The original experiment was developed in a Kaggle notebook.
These modules organize reusable code from the workflow. They may require
additional integration with the notebook's dataset paths and training setup
before the full pipeline can be executed independently.
