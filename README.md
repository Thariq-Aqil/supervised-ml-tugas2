
# Supervised Machine Learning - Task 2

## Image Classification

This project was developed for the Supervised Machine Learning course assignment. The objective is to classify document images into 16 classes using deep learning models.

## Competition

- **Platform:** Kaggle
- **Task:** Image Classification
- **Number of Classes:** 16
- **Best Kaggle Public Score:** 0.89418

## Models

### 1. DiT (BEiT)

A pretrained transformer-based model using `microsoft/dit-base`, fine-tuned for image classification.

### 2. ConvNeXt Tiny

A pretrained CNN model using `convnext_tiny.fb_in22k_ft_in1k`, fine-tuned for the same classification task.

### 3. Weighted Ensemble

The prediction probabilities from both models are combined using weighted averaging. The ensemble weight is selected based on validation accuracy.

## Workflow

1. Load training and testing metadata.
2. Resolve image paths and load TIFF images.
3. Convert images to grayscale and resize them to 224 × 224 pixels.
4. Cache image arrays for efficient processing.
5. Split the training data into training and validation sets using stratified sampling (94:6).
6. Apply data augmentation to training images.
7. Fine-tune the DiT model for 10 epochs.
8. Fine-tune ConvNeXt Tiny for 8 epochs.
9. Evaluate model performance on the validation set.
10. Optimize ensemble weights using validation accuracy.
11. Generate predictions for the test dataset.
12. Export predictions to `bismillah.csv`.

## Dataset

The dataset is not included in this repository. Obtain it from the Kaggle competition or the course-provided source; do not commit the dataset or generated checkpoints.

To use the dataset:

1. Download or accept the dataset from the competition/course source available to you.
2. Open `asdos-sml-tugas2-v2.ipynb` in Kaggle and choose **Add Input** to attach the dataset.
3. Confirm that the attached dataset contains `train.csv`, `test.csv`, and the image files referenced by the `filename` column.
4. The notebook searches under `/kaggle/input` for `train.csv`; the image paths in both CSV files must resolve beneath that dataset directory.
5. Keep the dataset outside the Git repository. The `.gitignore` excludes `data/`, common dataset image formats, CSV files, and checkpoint files.

## Repository Structure

```text
supervised-ml-tugas2/
├── README.md
├── .gitignore
├── requirements.txt
├── asdos-sml-tugas2-v2.ipynb
├── src/
│   ├── README.md
│   ├── config.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── dit_model.py
│   ├── convnext_model.py
│   ├── ensemble.py
│   └── evaluation.py
└── reports/
    └── results.md
```

## Requirements

The project uses the following libraries:

- PyTorch
- Torchvision
- Transformers
- timm
- NumPy
- Pandas
- Scikit-learn
- Pillow

The dependencies are listed in `requirements.txt`.

## How to Run

This project was developed using Kaggle Notebooks.

1. Open `asdos-sml-tugas2-v2.ipynb` in Kaggle.
2. Attach the required dataset using **Add Input**.
3. Enable GPU acceleration.
4. Install the required libraries if they are not already available.
5. Run the notebook cells sequentially.
6. The final prediction file will be generated as `bismillah.csv`.

## Results

The final ensemble submission achieved a Kaggle public score of **0.89418**.

The ensemble combines a transformer-based model and a CNN-based model. The weight is selected using accuracy on the held-out validation split, then the blended probabilities produce `bismillah.csv`. The reported public score is from the public leaderboard subset; it does not establish the private leaderboard result. Record the selected weight and validation metrics from the notebook output in [`reports/results.md`](reports/results.md) when available.

Further experiment details and interpretation are available in [`reports/results.md`](reports/results.md).

## Author

Supervised Machine Learning - Task 2  
Individual Assignment
