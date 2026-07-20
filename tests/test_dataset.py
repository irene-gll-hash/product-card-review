
from __future__ import annotations

import numpy as np

from card_checker.data.dataset import DEFECT_NAMES, ProductCardDataset
from card_checker.data.loaders import build_dataloaders, build_datasets
from card_checker.data.transforms import (
    build_test_transforms,
    build_train_transforms,
    build_validation_transforms,
)


def test_public_src_imports() -> None:
    assert callable(build_datasets)
    assert callable(build_dataloaders)
    assert callable(build_train_transforms)
    assert callable(build_validation_transforms)
    assert callable(build_test_transforms)


def test_dataset_converts_arrays_to_tensors() -> None:
    image = np.full((24, 32, 3), 127, dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=np.uint8)

    image_tensor = ProductCardDataset._to_image_tensor(image)
    mask_tensor = ProductCardDataset._to_mask_tensor(mask)

    assert image_tensor.shape == (3, 24, 32)
    assert mask_tensor.shape == (1, 24, 32)
    assert len(DEFECT_NAMES) == 6
