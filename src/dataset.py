
"""Dataset and DataLoader utilities for document image classification."""

import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset, DataLoader


class DocDS(Dataset):
    """Dataset gambar dokumen grayscale yang telah di-cache."""

    def __init__(self, X, y=None, train=False):
        """Initialize image arrays, labels, and optional augmentation."""
        self.X = X
        self.y = y
        self.train = train

        self.aug = T.RandomAffine(
            degrees=2,
            translate=(0.02, 0.02),
            scale=(0.96, 1.04),
            fill=255,
        )

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Return a normalized image and its label when available."""
        x = torch.from_numpy(self.X[i]).unsqueeze(0)

        if self.train:
            x = self.aug(x)

        x = ((x.float() / 255.0 - 0.5) / 0.5).expand(3, -1, -1)

        if self.y is not None:
            return x, int(self.y[i])

        return x


def make_dataloader(dataset, batch_size=64, shuffle=False):
    """Create a DataLoader with settings used in the notebook."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=shuffle,
    )
