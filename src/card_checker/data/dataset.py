"""Загрузка изображений, масок и меток датасета карточек товаров."""

from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

DEFECT_NAMES = ("blur", "skew", "rotated", "zoomed_out", "noise", "wrong_colors")
VALID_SPLITS = {"train", "validation", "test"}

Transform = Callable[..., dict[str, Any]]

class ProductCardDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        labels_file: str | Path,
        splits_file: str | Path,
        split: str,
        transform: Transform | None = None,
    ) -> None:
        self.labels_path = Path(labels_file).resolve()
        self.splits_path = Path(splits_file).resolve()
        self.data_root = self.labels_path.parent
        self.split = split
        self.transform = transform

        if not self.labels_path.is_file():
            raise FileNotFoundError(f"Файл разметки не найден: {self.labels_path}")
        if not self.splits_path.is_file():
            raise FileNotFoundError(f"Файл разделения не найден: {self.splits_path}")
        if split not in VALID_SPLITS:
            raise ValueError(f"split должен быть одним из {sorted(VALID_SPLITS)}, получено: {split}")

        labels = pd.read_csv(self.labels_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        splits = pd.read_csv(self.splits_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)

        required_label_columns = {
            "filename",
            "source_image",
            "card_id",
            "variant",
            "is_clean",
            "defect_count",
            "mask_source",
            "mask_filename",
        }
        required_label_columns.update(f"{name}_present" for name in DEFECT_NAMES)
        required_label_columns.update(f"{name}_intensity" for name in DEFECT_NAMES)

        missing_label_columns = required_label_columns - set(labels.columns)
        if missing_label_columns:
            raise ValueError(f"В labels.csv отсутствуют колонки: {sorted(missing_label_columns)}")

        required_split_columns = {"card_id", "split"}
        missing_split_columns = required_split_columns - set(splits.columns)
        if missing_split_columns:
            raise ValueError(f"В splits.csv отсутствуют колонки: {sorted(missing_split_columns)}")

        if labels.empty:
            raise ValueError("labels.csv не содержит строк")
        if splits.empty:
            raise ValueError("splits.csv не содержит строк")

        labels["card_id"] = labels["card_id"].str.strip()
        splits["card_id"] = splits["card_id"].str.strip()
        splits["split"] = splits["split"].str.strip()

        if labels["card_id"].eq("").any():
            raise ValueError("В labels.csv присутствуют пустые card_id")
        if splits["card_id"].eq("").any():
            raise ValueError("В splits.csv присутствуют пустые card_id")
        if splits["card_id"].duplicated().any():
            duplicated = sorted(splits.loc[splits["card_id"].duplicated(), "card_id"].unique())
            raise ValueError(f"В splits.csv повторяются card_id: {duplicated}")

        unknown_splits = sorted(set(splits["split"]) - VALID_SPLITS)
        if unknown_splits:
            raise ValueError(f"В splits.csv присутствуют неизвестные выборки: {unknown_splits}")

        samples = labels.merge(splits[["card_id", "split"]], on="card_id", how="left", validate="many_to_one")

        if samples["split"].isna().any():
            missing_ids = sorted(samples.loc[samples["split"].isna(), "card_id"].unique())
            raise ValueError(f"Для некоторых card_id отсутствует разделение: {missing_ids}")

        self.samples = samples.loc[samples["split"] == split].reset_index(drop=True)

        if self.samples.empty:
            raise ValueError(f"В выборке {split!r} нет изображений")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]

        image_path = self._resolve_path(row["filename"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")

        with Image.open(image_path) as image_file:
            image = np.array(image_file.convert("RGB"), copy=True)

        mask_filename = row["mask_filename"].strip()
        mask_available = bool(mask_filename)

        if mask_available:
            mask_path = self._resolve_path(mask_filename)
            if not mask_path.is_file():
                raise FileNotFoundError(f"Маска не найдена: {mask_path}")

            with Image.open(mask_path) as mask_file:
                mask = np.array(mask_file.convert("L"), copy=True)

            mask = (mask > 0).astype(np.uint8)
        else:
            mask_path = None
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Размер маски {mask.shape} не совпадает с размером изображения "
                f"{image.shape[:2]}: {image_path}"
            )

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)

            if "image" not in transformed:
                raise ValueError("Преобразование не вернуло ключ image")
            if "mask" not in transformed:
                raise ValueError("Преобразование не вернуло ключ mask")

            image = transformed["image"]
            mask = transformed["mask"]

        image_tensor = self._to_image_tensor(image)
        mask_tensor = self._to_mask_tensor(mask)

        target = torch.tensor(
            [float(row[f"{name}_present"]) for name in DEFECT_NAMES],
            dtype=torch.float32,
        )
        intensities = torch.tensor(
            [float(row[f"{name}_intensity"]) for name in DEFECT_NAMES],
            dtype=torch.float32,
        )

        return {
            "image": image_tensor,
            "target": target,
            "intensities": intensities,
            "mask": mask_tensor,
            "mask_available": torch.tensor(mask_available, dtype=torch.bool),
            "is_clean": torch.tensor(int(row["is_clean"]), dtype=torch.bool),
            "defect_count": torch.tensor(int(row["defect_count"]), dtype=torch.long),
            "card_id": row["card_id"],
            "variant": row["variant"],
            "filename": row["filename"],
            "source_image": row["source_image"],
            "mask_source": row["mask_source"],
            "image_path": str(image_path),
            "mask_path": "" if mask_path is None else str(mask_path),
        }

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.data_root / path).resolve()

    @staticmethod
    def _to_image_tensor(image: np.ndarray | Tensor) -> Tensor:
        if isinstance(image, Tensor):
            tensor = image
            if tensor.ndim != 3:
                raise ValueError(f"Изображение должно иметь три измерения, получено: {tuple(tensor.shape)}")
            if tensor.shape[0] not in {1, 3, 4} and tensor.shape[-1] in {1, 3, 4}:
                tensor = tensor.permute(2, 0, 1)
            if tensor.dtype == torch.uint8:
                tensor = tensor.float().div(255.0)
            else:
                tensor = tensor.float()
            return tensor.contiguous()

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Неподдерживаемый тип изображения: {type(image).__name__}")
        if image.ndim != 3:
            raise ValueError(f"Изображение должно иметь три измерения, получено: {image.shape}")

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        return tensor.div(255.0)

    @staticmethod
    def _to_mask_tensor(mask: np.ndarray | Tensor) -> Tensor:
        if isinstance(mask, Tensor):
            tensor = mask
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 3 and tensor.shape[0] != 1 and tensor.shape[-1] == 1:
                tensor = tensor.permute(2, 0, 1)
            if tensor.ndim != 3 or tensor.shape[0] != 1:
                raise ValueError(f"Маска должна иметь форму [1, H, W], получено: {tuple(tensor.shape)}")
            return (tensor > 0).float().contiguous()

        if not isinstance(mask, np.ndarray):
            raise TypeError(f"Неподдерживаемый тип маски: {type(mask).__name__}")
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise ValueError(f"Маска должна иметь два измерения, получено: {mask.shape}")

        tensor = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return (tensor > 0).float()