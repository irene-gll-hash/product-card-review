"""Генерация синтетического датасета дефектов фотографий товаров."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np

SEVERITY_MIN = 1
SEVERITY_MAX = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFECT_NAMES = ("blur", "skew", "rotated", "zoomed_out", "noise", "wrong_colors")
MASK_REQUIRED_DEFECTS = {"rotated", "zoomed_out"}

def _rand_severity() -> int:
    return random.randint(SEVERITY_MIN, SEVERITY_MAX)

def _check_severity(severity: int | None) -> int:
    value = _rand_severity() if severity is None else severity
    if not SEVERITY_MIN <= value <= SEVERITY_MAX:
        raise ValueError(f"severity должна быть от {SEVERITY_MIN} до {SEVERITY_MAX}, получено {value}")
    return value

def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)

def save_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise OSError(f"Не удалось закодировать изображение: {path}")
    try:
        encoded.tofile(path)
    except OSError as error:
        raise OSError(f"Не удалось сохранить изображение: {path}") from error

def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", mask)
    if not success:
        raise OSError(f"Не удалось закодировать маску: {path}")
    try:
        encoded.tofile(path)
    except OSError as error:
        raise OSError(f"Не удалось сохранить маску: {path}") from error

def _border_pixels(image: np.ndarray) -> np.ndarray:
    return np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0)

def _background_color(image: np.ndarray) -> tuple[int, int, int]:
    color = np.median(_border_pixels(image), axis=0).astype(np.uint8)
    return tuple(int(channel) for channel in color)

def _postprocess_mask(mask: np.ndarray) -> np.ndarray | None:
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None
    min_area = max(64, int(mask.size * 0.001))
    cleaned = np.zeros_like(mask)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == component] = 255
    ratio = float(np.count_nonzero(cleaned)) / cleaned.size
    if not 0.005 <= ratio <= 0.90:
        return None
    return cleaned

def _grabcut_mask(image: np.ndarray) -> np.ndarray | None:
    h, w = image.shape[:2]
    margin_x = max(1, int(w * 0.02))
    margin_y = max(1, int(h * 0.02))
    rect_w = w - 2 * margin_x
    rect_h = h - 2 * margin_y
    if rect_w < 2 or rect_h < 2:
        return None
    grabcut = np.zeros((h, w), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(image, grabcut, (margin_x, margin_y, rect_w, rect_h), bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    mask = np.where((grabcut == cv2.GC_FGD) | (grabcut == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _postprocess_mask(mask)

def estimate_product_mask(image: np.ndarray) -> np.ndarray | None:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = _border_pixels(lab)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    border_distance = np.linalg.norm(border - background, axis=1)
    threshold = max(15.0, float(np.percentile(border_distance, 95)) + 8.0)
    mask = _postprocess_mask((distance > threshold).astype(np.uint8) * 255)
    return mask if mask is not None else _grabcut_mask(image)

def load_product_mask(image_path: Path, input_dir: Path, masks_dir: Path | None, image: np.ndarray) -> tuple[np.ndarray | None, str]:
    if masks_dir is not None:
        relative = image_path.relative_to(input_dir)
        candidates = [masks_dir / relative.with_suffix(".png"), masks_dir / f"{image_path.stem}.png"]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            mask = read_image(candidate, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = _postprocess_mask(mask)
            if mask is not None:
                return mask, "file"
    return estimate_product_mask(image), "auto"

def _remove_product(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    border = _border_pixels(image).astype(np.float32)
    background_color = np.median(border, axis=0).astype(np.uint8)
    border_spread = float(np.mean(np.std(border, axis=0)))
    expanded_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    if border_spread < 18.0:
        background = image.copy()
        background[expanded_mask > 0] = background_color
        return background
    return cv2.inpaint(image, expanded_mask, 5, cv2.INPAINT_TELEA)

def _transform_product(image: np.ndarray, mask: np.ndarray, angle: float = 0.0, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Маска товара пуста")
    x, y, w, h = cv2.boundingRect(points)
    center = (x + w / 2.0, y + h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    background_color = _background_color(image)
    transformed_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background_color)
    transformed_mask = cv2.warpAffine(mask, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    background = _remove_product(image, mask)
    alpha = cv2.GaussianBlur(transformed_mask, (3, 3), 0).astype(np.float32)[:, :, None] / 255.0
    result = transformed_image.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8), transformed_mask

def add_blur(image: np.ndarray, severity: int | None = None, mask: np.ndarray | None = None) -> tuple[np.ndarray, int, np.ndarray | None]:
    severity = _check_severity(severity)
    kernel_sizes = {1: 7, 2: 21, 3: 51}
    max_kernel = min(image.shape[:2])
    max_kernel = max_kernel if max_kernel % 2 == 1 else max_kernel - 1
    kernel_size = min(kernel_sizes[severity], max_kernel)
    if kernel_size < 3:
        return image.copy(), severity, mask
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0), severity, mask

def add_skew(image: np.ndarray, severity: int | None = None, mask: np.ndarray | None = None) -> tuple[np.ndarray, int, np.ndarray | None]:
    severity = _check_severity(severity)
    h, w = image.shape[:2]
    shift = max(1, int(min(w, h) * 0.02 * severity))
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst = np.float32([[random.randint(0, shift), random.randint(0, shift)], [w - 1 - random.randint(0, shift), random.randint(0, shift)], [random.randint(0, shift), h - 1 - random.randint(0, shift)], [w - 1 - random.randint(0, shift), h - 1 - random.randint(0, shift)]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(image, matrix, (w, h), borderValue=_background_color(image))
    transformed_mask = cv2.warpPerspective(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) if mask is not None else None
    return result, severity, transformed_mask

def rotate_product(image: np.ndarray, severity: int | None = None, mask: np.ndarray | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    severity = _check_severity(severity)
    if mask is None:
        raise ValueError("Для rotated необходима маска товара")
    angle_ranges = {1: (10, 30), 2: (30, 100)}
    if severity == 3:
        angle = 180
    else:
        low, high = angle_ranges[severity]
        angle = random.choice((-1, 1)) * random.randint(low, high)
    result, transformed_mask = _transform_product(image, mask, angle=angle)
    return result, severity, transformed_mask

def zoom_out(image: np.ndarray, severity: int | None = None, mask: np.ndarray | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    severity = _check_severity(severity)
    if mask is None:
        raise ValueError("Для zoomed_out необходима маска товара")
    result, transformed_mask = _transform_product(image, mask, scale=1.0 - severity * 0.12)
    return result, severity, transformed_mask

def add_noise(image: np.ndarray, severity: int | None = None, mask: np.ndarray | None = None) -> tuple[np.ndarray, int, np.ndarray | None]:
    severity = _check_severity(severity)
    noise = np.random.normal(0, severity * 8, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy, severity, mask

def shift_colors(image: np.ndarray, severity: int | None = None, mask: np.ndarray | None = None) -> tuple[np.ndarray, int, np.ndarray | None]:
    severity = _check_severity(severity)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + severity * 12) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + severity * 0.12), 0, 255)
    shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return shifted, severity, mask

DEFECT_FUNCTIONS = {
    "blur": add_blur,
    "skew": add_skew,
    "rotated": rotate_product,
    "zoomed_out": zoom_out,
    "noise": add_noise,
    "wrong_colors": shift_colors,
}

def apply_combined_defects(image: np.ndarray, mask: np.ndarray | None, num_defects: int | None = None) -> tuple[np.ndarray, dict[str, int]]:
    available = [name for name in DEFECT_NAMES if mask is not None or name not in MASK_REQUIRED_DEFECTS]
    maximum = min(4, len(available))
    if maximum < 2:
        raise ValueError("Недостаточно доступных дефектов для комбинации")
    count = random.randint(2, maximum) if num_defects is None else num_defects
    if not 2 <= count <= maximum:
        raise ValueError(f"num_defects должна быть от 2 до {maximum}")
    chosen = random.sample(available, count)
    result = image.copy()
    current_mask = None if mask is None else mask.copy()
    applied: dict[str, int] = {}
    for defect_name in chosen:
        result, severity, current_mask = DEFECT_FUNCTIONS[defect_name](result, mask=current_mask)
        applied[defect_name] = severity
    return result, applied

def _image_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

def _card_id(image_path: Path, input_dir: Path) -> str:
    relative = image_path.relative_to(input_dir)
    return relative.parts[0] if len(relative.parts) > 1 else relative.stem

def _safe_stem(stem: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_-]+", "_", stem).strip("_")
    return safe or "image"

def _label_row(filename: str, source_image: str, card_id: str, variant: str, defects: dict[str, int], mask_source: str, mask_filename: str) -> dict[str, str | int]:
    row: dict[str, str | int] = {"filename": filename, "source_image": source_image, "card_id": card_id, "variant": variant, "is_clean": int(not defects), "mask_source": mask_source, "mask_filename": mask_filename}
    row.update({name: defects.get(name, 0) for name in DEFECT_NAMES})
    return row

def generate_dataset(input_dir: str | Path, output_dir: str | Path, combos_per_image: int = 5, seed: int = 42, masks_dir: str | Path | None = None, jpeg_quality: int = 95) -> None:
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    masks_path = Path(masks_dir).resolve() if masks_dir else None
    if not input_path.is_dir():
        raise NotADirectoryError(f"Папка с исходниками не найдена: {input_path}")
    if masks_path is not None and not masks_path.is_dir():
        raise NotADirectoryError(f"Папка с масками не найдена: {masks_path}")
    if combos_per_image < 0:
        raise ValueError("combos_per_image не может быть отрицательным")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality должна быть от 1 до 100")
    images = _image_paths(input_path)
    if not images:
        raise FileNotFoundError(f"В {input_path} не найдены изображения")
    images_dir = output_path / "images"
    generated_masks_dir = output_path / "masks"
    labels_path = output_path / "labels.csv"
    output_has_files = labels_path.exists() or (images_dir.exists() and any(images_dir.iterdir())) or (generated_masks_dir.exists() and any(generated_masks_dir.iterdir()))
    if output_has_files:
        raise FileExistsError(f"Папка результата уже содержит датасет: {output_path}. Укажите новую или пустую output_dir.")
    images_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    fieldnames = ["filename", "source_image", "card_id", "variant", "is_clean", "mask_source", "mask_filename", *DEFECT_NAMES]
    rows: list[dict[str, str | int]] = []
    skipped_images = 0
    skipped_geometry = 0
    for index, image_path in enumerate(images, start=1):
        image = read_image(image_path)
        if image is None:
            print(f"Пропуск: не удалось прочитать {image_path}")
            skipped_images += 1
            continue
        relative_source = image_path.relative_to(input_path).as_posix()
        card_id = _card_id(image_path, input_path)
        sample_id = f"{index:04d}_{_safe_stem(image_path.stem)}"
        mask, mask_source = load_product_mask(image_path, input_path, masks_path, image)
        if mask is None:
            mask_source = "missing"
            print(f"Предупреждение: маска не найдена для {relative_source}; rotated и zoomed_out будут пропущены.")
        mask_filename = ""
        if mask is not None:
            generated_mask_name = f"{sample_id}_mask.png"
            save_mask(generated_masks_dir / generated_mask_name, mask)
            mask_filename = f"masks/{generated_mask_name}"
        clean_name = f"{sample_id}_clean.jpg"
        save_jpeg(images_dir / clean_name, image, jpeg_quality)
        rows.append(_label_row(f"images/{clean_name}", relative_source, card_id, "clean", {}, mask_source, mask_filename))
        for defect_name in DEFECT_NAMES:
            if mask is None and defect_name in MASK_REQUIRED_DEFECTS:
                skipped_geometry += 1
                continue
            result, severity, _ = DEFECT_FUNCTIONS[defect_name](image.copy(), mask=None if mask is None else mask.copy())
            output_name = f"{sample_id}_{defect_name}.jpg"
            save_jpeg(images_dir / output_name, result, jpeg_quality)
            rows.append(_label_row(f"images/{output_name}", relative_source, card_id, "single", {defect_name: severity}, mask_source, mask_filename))
        for combo_index in range(1, combos_per_image + 1):
            result, applied = apply_combined_defects(image, mask)
            output_name = f"{sample_id}_combo_{combo_index}.jpg"
            save_jpeg(images_dir / output_name, result, jpeg_quality)
            rows.append(_label_row(f"images/{output_name}", relative_source, card_id, "combo", applied, mask_source, mask_filename))
    output_path.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Готово: {len(rows)} изображений и разметка {labels_path}")
    print(f"Seed: {seed}")
    if skipped_images:
        print(f"Не прочитано исходных изображений: {skipped_images}")
    if skipped_geometry:
        print(f"Пропущено геометрических версий без маски: {skipped_geometry}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация синтетического датасета дефектов фотографий товаров")
    parser.add_argument("--input_dir", required=True, help="Папка с корректными фотографиями")
    parser.add_argument("--output_dir", required=True, help="Новая или пустая папка результата")
    parser.add_argument("--masks_dir", default=None, help="Необязательная папка с бинарными масками товаров")
    parser.add_argument("--combos_per_image", type=int, default=5, help="Количество комбинированных версий на исходное фото")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимой генерации")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="Качество всех выходных JPEG")
    args = parser.parse_args()
    generate_dataset(args.input_dir, args.output_dir, args.combos_per_image, args.seed, args.masks_dir, args.jpeg_quality)
