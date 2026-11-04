
"""Utilities for loading and preprocessing document images."""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMG_SIZE = 224


def load_metadata(data_root):
    """Load training and testing metadata CSV files."""
    data_root = Path(data_root)
    train_df = pd.read_csv(data_root / "train.csv")
    test_df = pd.read_csv(data_root / "test.csv")
    return train_df, test_df


def resolve_image_path(data_root, relative_path):
    """Resolve an image path within the competition dataset directory."""
    data_root = Path(data_root)
    relative_path = Path(relative_path)

    candidates = [
        data_root / relative_path,
        data_root / relative_path.parts[0] / relative_path,
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(f"Image not found: {relative_path}")


def load_image(path, img_size=IMG_SIZE):
    """Load a TIFF image, convert it to grayscale, and resize it."""
    try:
        image = Image.open(path).convert("L")
        image = image.resize(
            (img_size, img_size),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(image, dtype=np.uint8)
    except Exception:
        return np.full((img_size, img_size), 255, dtype=np.uint8)


def cache_images(dataframe, data_root, img_size=IMG_SIZE):
    """Load and cache all images in a dataframe as a NumPy array."""
    paths = [
        resolve_image_path(data_root, filename)
        for filename in dataframe["filename"]
    ]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        images = list(
            executor.map(
                load_image,
                paths,
                [img_size] * len(paths),
                chunksize=64,
            )
        )

    return np.stack(images)
