"""Проверка изображений, масок и разметки синтетического датасета."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

DEFECT_NAMES = ("blur", "skew", "rotated", "zoomed_out", "noise", "wrong_colors")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MASK_SOURCES = {"auto", "file", "none"}
MASK_REQUIRED_DEFECTS = {"rotated", "zoomed_out"}
VARIANTS = {"clean", "single", "combo"}

def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)

def resolve_dataset_path(root: Path, relative_path: str) -> Path | None:
    if not relative_path:
        return None
    candidate = (root/relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate

def validate_dataset(dataset_dir: str | Path, good_images_dir: str | Path, labels_file: str | Path | None = None, max_messages: int = 50) -> bool:
    dataset_path = Path(dataset_dir).resolve()
    good_images_path = Path(good_images_dir).resolve()
    labels_path = Path(labels_file).resolve() if labels_file else dataset_path / "labels.csv"
    errors: list[str] = []
    warnings: list[str] = []

    def add_error(message:str) -> None:
        errors.append(message)
    def add_warning(message:str) -> None:
        warnings.append(message)

    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Папка датасета не найдена: {dataset_path}")
    if not good_images_path.is_dir():
        raise NotADirectoryError(f"Папка исходных изображений не найдена: {good_images_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Файл разметки не найден: {labels_path}")

    labels = pd.read_csv(labels_path, encoding="utf-8-sig", keep_default_na=False)
    if labels.empty:
        add_error("labels.csv empty")

    required_columns = [
        "filename",
        "source_image",
        "card_id",
        "variant",
        "is_clean",
        "defect_count",
        "mask_source",
        "mask_filename",
        "parameters_json",
    ]

    for defect_name in DEFECT_NAMES:
        required_columns.extend((f"{defect_name}_present", f"{defect_name}_intensity"))
    missing_columns = [column for column in required_columns if column not in labels.columns]
    if missing_columns:
        raise ValueError(f"В labels.csv отсутствуют колонки: {', '.join(missing_columns)}")
    numeric_columns = ["is_clean", "defect_count"]
    for defect_name in DEFECT_NAMES:
        numeric_columns.extend((f"{defect_name}_present", f"{defect_name}_intensity"))
    for column in numeric_columns:
        converted = pd.to_numeric(labels[column], errors="coerce")
        invalid_rows = converted[converted.isna()].index.tolist()
        for index in invalid_rows:
            add_error(f"Строка {index + 2}: колонка {column} содержит не число")
        labels[column] = converted

    duplicate_filenames = labels.loc[
        labels["filename"].duplicated(keep=False),
        "filename",
    ]
    for filename in sorted(duplicate_filenames.unique()):
        add_error(f"Повторяющийся filename: {filename}")
    empty_card_ids = labels.index[labels["card_id"].astype(str).str.strip() == ""].tolist()
    for index in empty_card_ids:
        add_error(f"Строка {index + 2}: пустой card_id")

    referenced_images: set[Path] = set()
    referenced_masks: set[Path] = set()

    for index, row in labels.iterrows():
        line = index + 2
        filename = str(row["filename"]).strip()
        source_image = str(row["source_image"]).strip()
        variant = str(row["variant"]).strip()
        mask_source = str(row["mask_source"])
        mask_filename = str(row["mask_filename"]).strip()

        if variant not in VARIANTS:
            add_error(f"Строка {line}: неизвестный variant={variant}")
        if mask_source not in MASK_SOURCES:
            add_error(f"Строка {line}: неизвестный mask_source={mask_source}")

        image_path = resolve_dataset_path(dataset_path, filename)
        image = None
        if image_path is None:
            add_error(f"Строка {line}: некорректный путь изображения: {filename}")
        elif not image_path.is_file():
            add_error(f"Строка {line}: изображение не найдено: {filename}")
        else:
            referenced_images.add(image_path)
            image = read_image(image_path)
            if image is None:
                add_error(f"Строка {line}: изображение не открывается: {filename}")
            elif image.ndim != 3 or image.shape[2] != 3:
                add_error(f"Строка {line}: изображение должно иметь три цветовых канала: {filename}")


        source_path = resolve_dataset_path(good_images_path, source_image)
        if source_path is None:
            add_error(f"Строка {line}: некорректный путь исходного изображения: {source_image}")
        elif not source_path.is_file():
            add_error(f"Строка {line}: исходное изображение не найдено: {source_image}")

        if mask_filename:
            mask_path = resolve_dataset_path(dataset_path, mask_filename)
            if mask_path is None:
                add_error(f"Строка {line}: некорректный путь маски: {mask_filename}")
            elif not mask_path.is_file():
                add_error(f"Строка {line}: маска не найдена: {mask_filename}")
            else:
                referenced_masks.add(mask_path)
                mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    add_error(f"Строка {line}: маска не открывается: {mask_filename}")
                else:
                    if image is not None and mask.shape != image.shape[:2]:
                        add_error(f"Строка {line}: размер маски {mask.shape} не совпадает с размером изображения {image.shape[:2]}")
                    unique_values = set(np.unique(mask).tolist())
                    if not unique_values.issubset({0, 255}):
                        add_warning(f"Строка {line}: маска содержит значения кроме 0 и 255: {mask_filename}")
                    foreground_ratio = float(np.count_nonzero(mask)) / mask.size
                    if foreground_ratio == 0:
                        add_error(f"Строка {line}: маска полностью пустая: {mask_filename}")
                    elif foreground_ratio < 0.005:
                        add_warning(f"Строка {line}: товар занимает меньше 0.5% маски: {mask_filename}")
                    elif foreground_ratio > 0.90:
                        add_warning(f"Строка {line}: товар занимает больше 90% маски: {mask_filename}")
            if mask_source == "none":
                add_error(f"Строка {line}: mask_source=none, но mask_filename заполнен")
        elif mask_source != "none":
            add_error(f"Строка {line}: для mask_source={mask_source} не указан mask_filename")

        present_defects: set[str] = set()
        for defect_name in DEFECT_NAMES:
            present_column = f"{defect_name}_present"
            intensity_column = f"{defect_name}_intensity"
            present = row[present_column]
            intensity = row[intensity_column]

            if pd.isna(present) or pd.isna(intensity):
                continue
            if present not in (0, 1):
                add_error(f"Строка {line}: {present_column} должен быть равен 0 или 1")
                continue
            if not 0.0 <= float(intensity) <= 1.0:
                add_error(f"Строка {line}: {intensity_column} должен находиться от 0 до 1")
            if present == 0 and float(intensity) != 0.0:
                add_error(f"Строка {line}: {present_column}=0, но {intensity_column}={intensity}")
            if present == 1 and float(intensity) <= 0.0:
                add_error(f"Строка {line}: {present_column}=1, но интенсивность не больше нуля")
            if present == 1:
                present_defects.add(defect_name)

        if present_defects & MASK_REQUIRED_DEFECTS and not mask_filename:
            add_error(
                f"Строка {line}: для rotated или zoomed_out необходима маска товара"
            )

        defect_count = row["defect_count"]
        is_clean = row["is_clean"]
        if not pd.isna(defect_count):
            if not float(defect_count).is_integer():
                add_error(f"Строка {line}: defect_count должен быть целым числом")
            elif int(defect_count) != len(present_defects):
                add_error(f"Строка {line}: defect_count={int(defect_count)}, фактически дефектов={len(present_defects)}")
        if not pd.isna(is_clean):
            if is_clean not in (0, 1):
                add_error(f"Строка {line}: is_clean должен быть равен 0 или 1")
            elif int(is_clean) != int(len(present_defects) == 0):
                add_error(f"Строка {line}: is_clean не соответствует наличию дефектов")

        if variant == "clean" and present_defects:
            add_error(f"Строка {line}: variant=clean содержит дефекты")
        if variant == "single" and len(present_defects) != 1:
            add_error(f"Строка {line}: variant=single должен содержать один дефект")
        if variant == "combo" and len(present_defects) < 2:
            add_error(f"Строка {line}: variant=combo должен содержать минимум два дефекта")

        try:
            parameters = json.loads(str(row["parameters_json"]))
        except json.JSONDecodeError:
            add_error(f"Строка {line}: некорректный parameters_json")
            continue
        if not isinstance(parameters, dict):
            add_error(f"Строка {line}: parameters_json должен содержать объект")
            continue

        application_order = parameters.get("application_order")
        parameter_defects = parameters.get("defects")
        if not isinstance(application_order, list):
            add_error(f"Строка {line}: application_order должен быть списком")
        if not isinstance(parameter_defects, dict):
            add_error(f"Строка {line}: defects внутри parameters_json должен быть объектом")
        else:
            parameter_names = set(parameter_defects)
            if parameter_names != present_defects:
                add_error(f"Строка {line}: дефекты в parameters_json не совпадают с колонками *_present")
        if isinstance(application_order, list) and set(application_order) != present_defects:
            add_error(f"Строка {line}: application_order не совпадает с обнаруженными дефектами")

    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"
    actual_images = {path.resolve() for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS} if images_dir.is_dir() else set()
    actual_masks = {path.resolve() for path in masks_dir.rglob("*.png") if path.is_file()} if masks_dir.is_dir() else set()

    for path in sorted(actual_images - referenced_images):
        add_warning(f"Изображение не указано в labels.csv: {path.relative_to(dataset_path)}")
    for path in sorted(actual_masks - referenced_masks):
        add_warning(f"Маска не указана в labels.csv: {path.relative_to(dataset_path)}")

    print(f"Строк в labels.csv: {len(labels)}")
    print(f"Уникальных card_id: {labels['card_id'].nunique()}")
    print(f"Clean: {(labels['variant'] == 'clean').sum()}")
    print(f"Single: {(labels['variant'] == 'single').sum()}")
    print(f"Combo: {(labels['variant'] == 'combo').sum()}")

    for defect_name in DEFECT_NAMES:
        count = int((labels[f"{defect_name}_present"] == 1).sum())
        print(f"{defect_name}: {count}")

    if warnings:
        print(f"\nПредупреждений: {len(warnings)}")
        for message in warnings[:max_messages]:
            print(f"WARNING: {message}")
        if len(warnings) > max_messages:
            print(f"Показаны первые {max_messages} предупреждений")

    if errors:
        print(f"\nОшибок: {len(errors)}")
        for message in errors[:max_messages]:
            print(f"ERROR: {message}")
        if len(errors) > max_messages:
            print(f"Показаны первые {max_messages} ошибок")
        print("\nДатасет не прошёл проверку")
        return False

    print("\nДатасет прошёл проверку")
    return True

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка синтетического датасета")
    parser.add_argument("--dataset_dir", default="data/synthetic", help="Папка синтетического датасета")
    parser.add_argument("--good_images_dir", default="data/raw/good_images", help="Папка исходных изображений")
    parser.add_argument("--labels_file", default=None, help="Путь к labels.csv")
    parser.add_argument("--max_messages", type=int, default=50, help="Максимальное количество выводимых ошибок и предупреждений")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    is_valid = validate_dataset(args.dataset_dir, args.good_images_dir, args.labels_file, args.max_messages)
    raise SystemExit(0 if is_valid else 1)

if __name__ == "__main__":
    main()
