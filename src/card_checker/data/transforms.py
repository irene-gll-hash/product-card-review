"""Преобразования изображений и масок для обучения и оценки моделей."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResizeAndPad:
    """Изменяет размер с сохранением пропорций и дополняет изображение до квадрата."""

    def __init__(self, size: int, image_fill: int = 255, mask_fill: int = 0) -> None:
        if size <= 0:
            raise ValueError(f"size должен быть положительным, получено: {size}")

        self.size = size
        self.image_fill = image_fill
        self.mask_fill = mask_fill

    def __call__(
        self,
        image: tv_tensors.Image,
        mask: tv_tensors.Mask,
    ) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        height, width = image.shape[-2:]
        scale = self.size / max(height, width)

        new_height = max(1, round(height * scale))
        new_width = max(1, round(width * scale))

        image = F.resize(
            image,
            size=[new_height, new_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = F.resize(
            mask,
            size=[new_height, new_width],
            interpolation=InterpolationMode.NEAREST,
        )

        padding_height = self.size - new_height
        padding_width = self.size - new_width

        left = padding_width // 2
        right = padding_width - left
        top = padding_height // 2
        bottom = padding_height - top
        padding = [left, top, right, bottom]

        image = F.pad(image, padding=padding, fill=self.image_fill)
        mask = F.pad(mask, padding=padding, fill=self.mask_fill)

        return image, mask


class ProductCardTransforms:
    """Применяет одинаковые геометрические преобразования к изображению и маске."""

    def __init__(
        self,
        image_size: int = 256,
        train: bool = False,
        horizontal_flip_probability: float = 0.5,
        normalize: bool = True,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError(
                "horizontal_flip_probability должен находиться в диапазоне от 0 до 1"
            )
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean и std должны содержать по три значения для RGB")
        if any(value <= 0 for value in std):
            raise ValueError("Все значения std должны быть положительными")

        self.train = train
        self.normalize = normalize
        self.resize_and_pad = ResizeAndPad(image_size)
        self.horizontal_flip = v2.RandomHorizontalFlip(
            p=horizontal_flip_probability
        )
        self.to_float = v2.ToDtype(torch.float32, scale=True)
        self.normalize_image = v2.Normalize(
            mean=list(mean),
            std=list(std),
        )

    def __call__(self, *, image: np.ndarray,  mask: np.ndarray) -> dict[str, Any]:
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image должен быть массивом NumPy, получено: {type(image).__name__}"
            )
        if not isinstance(mask, np.ndarray):
            raise TypeError(
                f"mask должна быть массивом NumPy, получено: {type(mask).__name__}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Изображение должно иметь форму [H, W, 3], получено: {image.shape}"
            )
        if mask.ndim != 2:
            raise ValueError(
                f"Маска должна иметь форму [H, W], получено: {mask.shape}"
            )
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Размер изображения {image.shape[:2]} не совпадает "
                f"с размером маски {mask.shape}"
            )

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image)
        ).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask)
        )

        image_tensor = tv_tensors.Image(image_tensor)
        mask_tensor = tv_tensors.Mask(mask_tensor)

        image_tensor, mask_tensor = self.resize_and_pad(
            image_tensor,
            mask_tensor,
        )

        if self.train:
            image_tensor, mask_tensor = self.horizontal_flip(
                image_tensor,
                mask_tensor,
            )

        image_tensor = self.to_float(image_tensor)
        mask_tensor = (mask_tensor > 0).to(torch.float32)

        if self.normalize:
            image_tensor = self.normalize_image(image_tensor)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
        }


def build_train_transforms(
    image_size: int = 256,
    horizontal_flip_probability: float = 0.5,
    normalize: bool = True,
) -> ProductCardTransforms:
    return ProductCardTransforms(
        image_size=image_size,
        train=True,
        horizontal_flip_probability=horizontal_flip_probability,
        normalize=normalize,
    )


def build_validation_transforms(
    image_size: int = 256,
    normalize: bool = True,
) -> ProductCardTransforms:
    return ProductCardTransforms(
        image_size=image_size,
        train=False,
        horizontal_flip_probability=0.0,
        normalize=normalize,
    )


def build_test_transforms(
        image_size: int = 256,
        normalize: bool = True,
) -> ProductCardTransforms:
    return build_validation_transforms(
        image_size=image_size,
        normalize=normalize,
    )