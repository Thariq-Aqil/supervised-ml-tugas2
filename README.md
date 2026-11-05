
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

The dataset is not included in this repository because it is provided through Kaggle and may be large.

To use the dataset:

1. Download or access the dataset from the official Kaggle competition or course-provided source.
2. Open the notebook in Kaggle.
3. Add the dataset through the **Add Input** option.
4. Ensure the dataset includes the required metadata files (`train.csv` and `test.csv`) and image files.
5. Confirm that the notebook can access the dataset before running the cells.

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

The ensemble combines transformer-based and CNN-based architectures to leverage their different feature representations. The ensemble weight is selected based on validation accuracy. The public score reflects performance on the public evaluation subset; the private leaderboard score may differ.

Further experiment details and interpretation are available in [`reports/results.md`](reports/results.md).

## Author

Supervised Machine Learning - Task 2  
Individual Assignment
