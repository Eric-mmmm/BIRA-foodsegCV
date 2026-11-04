"""Shared interface for every BIRA food-segmentation model.

The ensemble, the custom (from-scratch) model and the evaluation pipeline all
talk to each other only through the two things defined here:

    Detection   - what ONE food item looks like when a model reports it
    FoodModel   - the methods every model must provide

Changes to this file need a review from all three owners (see CODEOWNERS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Detection:
    """One detected food item in an image.

    Pixel coordinates use image convention: x = column (left -> right),
    y = row (top -> bottom), origin at the top-left corner.
    """

    label: str
    """Food class name, e.g. "rice". Must come from the shared class list."""

    score: float
    """Model confidence in [0, 1]."""

    mask: np.ndarray
    """Boolean array of shape (H, W). True where the item is. Same H, W as the input image."""

    bbox: tuple[int, int, int, int]
    """Tight box around the mask as (x_min, y_min, x_max, y_max), in pixels."""

    centroid_px: tuple[int, int]
    """Target point for the arm as (x, y) in pixels. Must lie inside the mask."""


@runtime_checkable
class FoodModel(Protocol):
    """Anything that can look at a plate and list the food on it."""

    name: str
    """Short identifier used in result files, e.g. "custom_unet"."""

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Find every food item in one image.

        Args:
            image: RGB image, uint8, shape (H, W, 3).

        Returns:
            One Detection per food item, in any order. Empty list if no food.
        """
        ...
