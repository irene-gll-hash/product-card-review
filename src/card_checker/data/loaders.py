"""Создание датасетов и загрузчиков для train, validation и test."""

from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from card_checker.data.dataset import ProductCardDataset
from card_checker.data.transforms import (
    build_test_transforms,
    build_train_transforms,
    build_validation_transforms,
)


def build_datasets(
    labels_file: str | Path,
    splits_file: str | Path,
    image_size: int = 256,
    horizontal_flip_probability: float = 0.5,
    normalize: bool = True,
) -> dict[str, ProductCardDataset]:
    """Создаёт датасеты для обучающей, валидационной и тестовой выборок."""

    train_dataset = ProductCardDataset(
        labels_file=labels_file,
        splits_file=splits_file,
        split="train",
        transform=build_train_transforms(
            image_size=image_size,
            horizontal_flip_probability=horizontal_flip_probability,
            normalize=normalize,
        ),
    )

    validation_dataset = ProductCardDataset(
        labels_file=labels_file,
        splits_file=splits_file,
        split="validation",
        transform=build_validation_transforms(
            image_size=image_size,
            normalize=normalize,
        ),
    )

    test_dataset = ProductCardDataset(
        labels_file=labels_file,
        splits_file=splits_file,
        split="test",
        transform=build_test_transforms(
            image_size=image_size,
            normalize=normalize,
        ),
    )

    return {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }


def build_dataloaders(
    labels_file: str | Path,
    splits_file: str | Path,
    image_size: int = 256,
    batch_size: int = 16,
    horizontal_flip_probability: float = 0.5,
    normalize: bool = True,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Создаёт DataLoader для обучающей, валидационной и тестовой выборок."""

    if batch_size <= 0:
        raise ValueError(
            f"batch_size должен быть положительным, получено: {batch_size}"
        )

    if num_workers < 0:
        raise ValueError(
            f"num_workers не может быть отрицательным, получено: {num_workers}"
        )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    datasets = build_datasets(
        labels_file=labels_file,
        splits_file=splits_file,
        image_size=image_size,
        horizontal_flip_probability=horizontal_flip_probability,
        normalize=normalize,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        dataset=datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        generator=generator,
    )

    validation_loader = DataLoader(
        dataset=datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        dataset=datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }