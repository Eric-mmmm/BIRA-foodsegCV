"""FoodSeg103 loader (Hugging Face version: EduardoPacheco/FoodSeg103).

What the data looks like
    7,118 plated-food photos with pixel-wise masks. Each mask pixel stores a
    class id: 0 = background, 1..102 = ingredients, 103 = "other ingredients"
    (104 values in total).

Splits
    Hugging Face only ships "train" (4,983 images) and "validation" (2,135).
    We need three sets, so we use:
        train -> 90% of the official train split (seeded shuffle)
        val   -> the other 10%, for picking hyperparameters / the best epoch
        test  -> the official validation split, never used while tuning
    Split files hold row positions inside the Hugging Face split they refer to
    (train/val -> HF "train", test -> HF "validation") and are written to
    splits/ so the whole team can reuse the exact same lists.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

HF_NAME = "EduardoPacheco/FoodSeg103"
NUM_CLASSES = 104  # background + 103 ingredient classes
BACKGROUND = 0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_split_ids(n_total: int, val_fraction: float = 0.1, seed: int = 0):
    """Shuffle 0..n_total-1 with a fixed seed and cut it into (train_ids, val_ids)."""
    ids = list(range(n_total))
    random.Random(seed).shuffle(ids)
    n_val = int(round(n_total * val_fraction))
    return sorted(ids[n_val:]), sorted(ids[:n_val])


def save_ids(ids, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(i) for i in ids) + "\n")


def load_ids(path: str | Path) -> list[int]:
    return [int(line) for line in Path(path).read_text().split()]


class FoodSeg103(Dataset):
    """One item = (image, mask).

    image: float32 tensor (3, S, S), ImageNet-normalized
    mask : int64 tensor   (S, S), values in 0..103

    NOTE: images are resized to a square S x S, which squashes the aspect ratio.
    That is fine for a first model; we can switch to padding later.
    """

    def __init__(self, hf_split, indices=None, image_size: int = 256, train: bool = False):
        self.ds = hf_split
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)
        size = (image_size, image_size)

        steps = []
        if train:
            steps += [
                v2.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # image only
            ]
        else:
            steps += [v2.Resize(size)]
        steps += [
            v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # only touches the image
        ]
        self.transform = v2.Compose(steps)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = self.ds[self.indices[i]]
        image = np.array(row["image"].convert("RGB"))          # (H, W, 3) uint8
        mask = np.array(row["label"])                          # (H, W) class ids
        if mask.ndim == 3:                                     # some exports store 3 identical channels
            mask = mask[..., 0]
        image = tv_tensors.Image(torch.from_numpy(image).permute(2, 0, 1))
        mask = tv_tensors.Mask(torch.from_numpy(mask.astype(np.uint8)))
        image, mask = self.transform(image, mask)
        return image, mask


def build_datasets(image_size: int = 256, val_fraction: float = 0.1, seed: int = 0,
                   splits_dir: str | Path = "splits"):
    """Download (first run) and return (train_set, val_set, test_set); also writes the split files."""
    from datasets import load_dataset  # imported here so the rest of the file works without it

    hf = load_dataset(HF_NAME)
    train_ids, val_ids = make_split_ids(len(hf["train"]), val_fraction, seed)
    test_ids = list(range(len(hf["validation"])))

    splits_dir = Path(splits_dir)
    save_ids(train_ids, splits_dir / "foodseg103_train.txt")
    save_ids(val_ids, splits_dir / "foodseg103_val.txt")
    save_ids(test_ids, splits_dir / "foodseg103_test.txt")

    return (
        FoodSeg103(hf["train"], train_ids, image_size, train=True),
        FoodSeg103(hf["train"], val_ids, image_size, train=False),
        FoodSeg103(hf["validation"], test_ids, image_size, train=False),
    )
