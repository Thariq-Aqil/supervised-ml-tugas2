# Experiment Results

## Model Overview

This project evaluates two pretrained deep learning architectures for image classification:

- **DiT (BEiT):** Transformer-based model fine-tuned for 10 epochs.
- **ConvNeXt Tiny:** CNN-based model fine-tuned for 8 epochs.
- **Weighted Ensemble:** Combines prediction probabilities from both models.

## Validation Strategy

The training dataset is split into:

- 94% training data
- 6% validation data

The split uses stratified sampling with a fixed random seed of 42 to preserve class distribution.

The ensemble weight is selected based on validation accuracy by testing different weights between the DiT and ConvNeXt predictions.

## Kaggle Result

| Metric | Result |
|---|---:|
| Best Kaggle Score | 0.89418 |
| Submission File | `bismillah.csv` |
| Prediction Method | Weighted Ensemble |

## Interpretation

The final ensemble combines the prediction probabilities of a transformer-based model and a CNN-based model.

The ensemble weight is selected using validation accuracy, allowing the contribution of each model to be adjusted based on its performance on the validation set.

The final predictions are generated for the test dataset and exported to `bismillah.csv` for submission to Kaggle.

## Notes

Detailed validation accuracy for each individual model and the selected ensemble weight should be recorded from the notebook output.
