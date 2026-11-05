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

The final ensemble combines the prediction probabilities of a transformer-based model and a CNN-based model. The models provide two different architecture families, and validation accuracy is used to choose how much each model contributes to the blended probabilities.

The Kaggle public score of **0.89418** is the observed score on the public leaderboard subset. It should not be treated as the private leaderboard score or as a guarantee of generalization. The repository does not currently record per-model validation metrics or the selected ensemble weight, so no claim about which model contributed more is made here.

The final predictions are generated for the test dataset and exported to `bismillah.csv` for submission to Kaggle.

## Validation details to record

After a run, record the DiT validation accuracy, ConvNeXt validation accuracy, selected DiT ensemble weight, and ensemble validation accuracy from the notebook output. This makes the model comparison and interpretation reproducible without inventing values that are not currently captured in the report.
