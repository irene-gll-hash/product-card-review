"""Генерация разнообразного синтетического датасета дефектов карточек товаров."""
from __future__ import annotations
import argparse
import csv
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any
import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFECT_NAMES = ("blur", "skew", "rotated", "zoomed_out", "noise", "wrong_colors")
MASK_REQUIRED_DEFECTS = {"rotated", "zoomed_out"}

def sample_intensity() -> float:
    low, high = random.choice(((0.05, 0.35), (0.35, 0.70), (0.70, 1.00)))
    return random.uniform(low, high)

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
    encoded.tofile(path)

def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", mask)
    if not success:
        raise OSError(f"Не удалось закодировать маску: {path}")
    encoded.tofile(path)

def border_pixels(image: np.ndarray) -> np.ndarray:
    return np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0)

def background_color(image: np.ndarray) -> tuple[int, int, int]:
    color = np.median(border_pixels(image), axis=0).astype(np.uint8)
    return tuple(int(channel) for channel in color)

def postprocess_mask(mask: np.ndarray) -> np.ndarray | None:
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None
    minimum_area = max(64, int(mask.size * 0.001))
    cleaned = np.zeros_like(mask)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == component] = 255
    mask_ratio = float(np.count_nonzero(cleaned)) / cleaned.size
    if not 0.005 <= mask_ratio <= 0.90:
        return None
    return cleaned

def grabcut_mask(image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    margin_x = max(1, int(width * 0.02))
    margin_y = max(1, int(height * 0.02))
    rectangle_width = width - 2 * margin_x
    rectangle_height = height - 2 * margin_y
    if rectangle_width < 2 or rectangle_height < 2:
        return None
    grabcut = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(image, grabcut, (margin_x, margin_y, rectangle_width, rectangle_height), background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    mask = np.where((grabcut == cv2.GC_FGD) | (grabcut == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return postprocess_mask(mask)

def estimate_product_mask(image: np.ndarray) -> np.ndarray | None:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = border_pixels(lab)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    border_distance = np.linalg.norm(border - background, axis=1)
    threshold = max(15.0, float(np.percentile(border_distance, 95)) + 8.0)
    mask = postprocess_mask((distance > threshold).astype(np.uint8) * 255)
    return mask if mask is not None else grabcut_mask(image)

def load_product_mask(image_path: Path, input_dir: Path, masks_dir: Path | None, image: np.ndarray) -> tuple[np.ndarray | None, str]:
    if masks_dir is not None:
        relative_path = image_path.relative_to(input_dir)
        candidates = [masks_dir / relative_path.with_suffix(".png"), masks_dir / f"{image_path.stem}.png"]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            mask = read_image(candidate, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = postprocess_mask(mask)
            if mask is not None:
                return mask, "file"
    mask = estimate_product_mask(image)
    return (mask, "auto") if mask is not None else (None, "none")

def remove_product(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    border = border_pixels(image).astype(np.float32)
    color = np.median(border, axis=0).astype(np.uint8)
    border_spread = float(np.mean(np.std(border, axis=0)))
    expanded_mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    if border_spread < 18.0:
        background = image.copy()
        background[expanded_mask > 0] = color
        return background
    return cv2.inpaint(image, expanded_mask, 5, cv2.INPAINT_TELEA)

def transform_product(image: np.ndarray, mask: np.ndarray, angle: float = 0.0, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Маска товара пуста")
    x, y, width, height = cv2.boundingRect(points)
    center = (x + width / 2.0, y + height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    transformed_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background_color(image))
    transformed_mask = cv2.warpAffine(mask, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    background = remove_product(image, mask)
    alpha = cv2.GaussianBlur(transformed_mask, (3, 3), 0).astype(np.float32)[:, :, None] / 255.0
    result = transformed_image.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8), transformed_mask

def odd_kernel_size(value: int, image: np.ndarray) -> int:
    maximum = min(image.shape[:2])
    if maximum % 2 == 0:
        maximum -= 1
    if maximum < 3:
        return 1
    value = value if value % 2 == 1 else value + 1
    return max(3, min(value, maximum))

def gaussian_blur(image: np.ndarray, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    sigma = 0.5 + intensity * 9.5
    kernel_size = odd_kernel_size(int(5 + intensity * 56), image)
    result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    return result, {"type": "gaussian", "kernel_size": kernel_size, "sigma": round(sigma, 4)}

def motion_blur(image: np.ndarray, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    kernel_size = odd_kernel_size(int(5 + intensity * 48), image)
    angle = random.uniform(0.0, 180.0)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2.0
    radius = center
    radians = np.deg2rad(angle)
    x_shift = np.cos(radians) * radius
    y_shift = np.sin(radians) * radius
    start = (int(round(center - x_shift)), int(round(center - y_shift)))
    end = (int(round(center + x_shift)), int(round(center + y_shift)))
    cv2.line(kernel, start, end, 1.0, 1)
    kernel_sum = float(kernel.sum())
    if kernel_sum == 0:
        kernel[int(center), int(center)] = 1.0
        kernel_sum = 1.0
    kernel /= kernel_sum
    result = cv2.filter2D(image, -1, kernel)
    return result, {"type": "motion", "kernel_size": kernel_size, "angle": round(angle, 4)}

def defocus_blur(image: np.ndarray, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    kernel_size = odd_kernel_size(int(5 + intensity * 36), image)
    radius = max(1, kernel_size // 2)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    kernel /= float(kernel.sum())
    result = cv2.filter2D(image, -1, kernel)
    return result, {"type": "defocus", "kernel_size": kernel_size, "radius": radius}

def add_blur(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None, float, dict[str, Any]]:
    intensity = sample_intensity()
    blur_type = random.choice(("gaussian", "motion", "defocus"))
    if blur_type == "gaussian":
        result, parameters = gaussian_blur(image, intensity)
    elif blur_type == "motion":
        result, parameters = motion_blur(image, intensity)
    else:
        result, parameters = defocus_blur(image, intensity)
    return result, mask, intensity, parameters

def add_skew(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None, float, dict[str, Any]]:
    intensity = sample_intensity()
    height, width = image.shape[:2]
    maximum_shift = max(1.0, min(width, height) * (0.01 + intensity * 0.16))
    source = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    destination = np.float32([
        [random.uniform(0, maximum_shift), random.uniform(0, maximum_shift)],
        [width - 1 - random.uniform(0, maximum_shift), random.uniform(0, maximum_shift)],
        [random.uniform(0, maximum_shift), height - 1 - random.uniform(0, maximum_shift)],
        [width - 1 - random.uniform(0, maximum_shift), height - 1 - random.uniform(0, maximum_shift)],
    ])
    matrix = cv2.getPerspectiveTransform(source, destination)
    result = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background_color(image))
    transformed_mask = cv2.warpPerspective(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) if mask is not None else None
    points = [[round(float(x), 3), round(float(y), 3)] for x, y in destination]
    parameters = {"maximum_shift_pixels": round(maximum_shift, 4), "destination_points": points}
    return result, transformed_mask, intensity, parameters

def add_rotation(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    if mask is None:
        raise ValueError("Для rotated необходима маска товара")
    intensity = sample_intensity()
    anchor_probability = random.random()
    if anchor_probability < 0.12:
        angle = 180.0
    elif anchor_probability < 0.24:
        angle = random.choice((-90.0, 90.0))
    else:
        angle = random.choice((-1.0, 1.0)) * (5.0 + intensity * 175.0)
    intensity = min(1.0, abs(angle) / 180.0)
    result, transformed_mask = transform_product(image, mask, angle=angle)
    return result, transformed_mask, intensity, {"angle_degrees": round(angle, 4)}

def add_zoom_out(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    if mask is None:
        raise ValueError("Для zoomed_out необходима маска товара")
    intensity = sample_intensity()
    scale = 0.92 - intensity * 0.57
    result, transformed_mask = transform_product(image, mask, scale=scale)
    return result, transformed_mask, intensity, {"scale": round(scale, 4)}

def gaussian_noise(image: np.ndarray, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    standard_deviation = 2.0 + intensity * 38.0
    noise = np.random.normal(0.0, standard_deviation, image.shape).astype(np.float32)
    result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return result, {"type": "gaussian", "standard_deviation": round(standard_deviation, 4)}

def salt_pepper_noise(image: np.ndarray, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    amount = 0.001 + intensity * 0.05
    random_map = np.random.random(image.shape[:2])
    result = image.copy()
    result[random_map < amount / 2.0] = 0
    result[random_map > 1.0 - amount / 2.0] = 255
    return result, {"type": "salt_pepper", "amount": round(amount, 6)}

def poisson_noise(image: np.ndarray, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    poisson_level = 60.0 - intensity * 55.0
    normalized = image.astype(np.float32) / 255.0
    result = np.random.poisson(normalized * poisson_level) / poisson_level
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result, {"type": "poisson", "poisson_level": round(poisson_level, 4)}

def jpeg_noise(image: np.ndarray, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    quality = max(15, int(round(95 - intensity * 80)))
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise OSError("Не удалось создать JPEG-артефакты")
    result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return result, {"type": "jpeg", "quality": quality}

def add_noise(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None, float, dict[str, Any]]:
    intensity = sample_intensity()
    noise_type = random.choice(("gaussian", "salt_pepper", "poisson", "jpeg"))
    if noise_type == "gaussian":
        result, parameters = gaussian_noise(image, intensity)
    elif noise_type == "salt_pepper":
        result, parameters = salt_pepper_noise(image, intensity)
    elif noise_type == "poisson":
        result, parameters = poisson_noise(image, intensity)
    else:
        result, parameters = jpeg_noise(image, intensity)
    return result, mask, intensity, parameters

def apply_hue(image: np.ndarray, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    shift = random.choice((-1.0, 1.0)) * (3.0 + intensity * 42.0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result, {"operation": "hue", "shift": round(shift, 4)}

def apply_saturation(image: np.ndarray, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    change = 0.08 + intensity * 0.82
    factor = 1.0 + change if random.random() < 0.5 else max(0.1, 1.0 - change)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result, {"operation": "saturation", "factor": round(factor, 4)}

def apply_brightness(image: np.ndarray, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    offset = random.choice((-1.0, 1.0)) * (4.0 + intensity * 76.0)
    result = np.clip(image.astype(np.float32) + offset, 0, 255).astype(np.uint8)
    return result, {"operation": "brightness", "offset": round(offset, 4)}

def apply_contrast(image: np.ndarray, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    change = 0.05 + intensity * 0.65
    factor = 1.0 + change if random.random() < 0.5 else max(0.25, 1.0 - change)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    result = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    return result, {"operation": "contrast", "factor": round(factor, 4)}

def apply_temperature(image: np.ndarray, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    shift = random.choice((-1.0, 1.0)) * (4.0 + intensity * 66.0)
    result = image.astype(np.float32)
    result[:, :, 2] += shift
    result[:, :, 0] -= shift
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result, {"operation": "temperature", "shift": round(shift, 4)}

COLOR_FUNCTIONS = {
    "hue": apply_hue,
    "saturation": apply_saturation,
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "temperature": apply_temperature,
}

def add_wrong_colors(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None, float, dict[str, Any]]:
    intensity = sample_intensity()
    operation_names = random.sample(list(COLOR_FUNCTIONS), random.randint(1, 3))
    result = image.copy()
    operations: list[dict[str, Any]] = []
    for operation_name in operation_names:
        result, parameters = COLOR_FUNCTIONS[operation_name](result, intensity)
        operations.append(parameters)
    return result, mask, intensity, {"operations": operations}

DEFECT_FUNCTIONS = {
    "blur": add_blur,
    "skew": add_skew,
    "rotated": add_rotation,
    "zoomed_out": add_zoom_out,
    "noise": add_noise,
    "wrong_colors": add_wrong_colors,
}

def apply_defects(image: np.ndarray, mask: np.ndarray | None, defect_names: list[str]) -> tuple[np.ndarray, np.ndarray | None, dict[str, dict[str, Any]], list[str]]:
    result = image.copy()
    current_mask = None if mask is None else mask.copy()
    order = defect_names.copy()
    random.shuffle(order)
    applied: dict[str, dict[str, Any]] = {}
    for defect_name in order:
        result, current_mask, intensity, parameters = DEFECT_FUNCTIONS[defect_name](result, current_mask)
        applied[defect_name] = {"intensity": round(float(intensity), 6), "parameters": parameters}
    return result, current_mask, applied, order

def image_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

def get_card_id(image_path: Path, input_dir: Path) -> str:
    relative_path = image_path.relative_to(input_dir)
    return relative_path.parts[0] if len(relative_path.parts) > 1 else relative_path.stem

def safe_stem(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_-]+", "_", value).strip("_")
    return result or "image"

def create_label_row(filename: str, source_image: str, card_id: str, variant: str, mask_source: str, mask_filename: str, applied: dict[str, dict[str, Any]], order: list[str]) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "filename": filename,
        "source_image": source_image,
        "card_id": card_id,
        "variant": variant,
        "is_clean": int(not applied),
        "defect_count": len(applied),
        "mask_source": mask_source,
        "mask_filename": mask_filename,
    }
    for defect_name in DEFECT_NAMES:
        row[f"{defect_name}_present"] = int(defect_name in applied)
        row[f"{defect_name}_intensity"] = applied.get(defect_name, {}).get("intensity", 0.0)
    parameters = {"application_order": order, "defects": {name: value["parameters"] for name, value in applied.items()}}
    row["parameters_json"] = json.dumps(parameters, ensure_ascii=False, separators=(",", ":"))
    return row

def generate_dataset(input_dir: str | Path, output_dir: str | Path, masks_dir: str | Path | None = None, singles_per_defect: int = 2, combos_per_image: int = 8, minimum_combo_defects: int = 2, maximum_combo_defects: int = 4, seed: int = 42, jpeg_quality: int = 95, overwrite: bool = False) -> None:
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    masks_path = Path(masks_dir).resolve() if masks_dir else None
    if not input_path.is_dir():
        raise NotADirectoryError(f"Папка с исходными изображениями не найдена: {input_path}")
    if masks_path is not None and not masks_path.is_dir():
        raise NotADirectoryError(f"Папка с масками не найдена: {masks_path}")
    if input_path == output_path or input_path in output_path.parents:
        raise ValueError("Папка результата не должна находиться внутри папки с исходными изображениями")
    if singles_per_defect < 1:
        raise ValueError("singles_per_defect должна быть не меньше 1")
    if combos_per_image < 0:
        raise ValueError("combos_per_image не может быть отрицательным")
    if not 2 <= minimum_combo_defects <= maximum_combo_defects:
        raise ValueError("Неверные границы количества комбинированных дефектов")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality должна быть от 1 до 100")
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Папка результата не пуста: {output_path}. Используй --overwrite для перезаписи")
        shutil.rmtree(output_path)
    random.seed(seed)
    np.random.seed(seed)
    images_output = output_path / "images"
    masks_output = output_path / "masks"
    images_output.mkdir(parents=True, exist_ok=True)
    masks_output.mkdir(parents=True, exist_ok=True)
    sources = image_paths(input_path)
    if not sources:
        raise FileNotFoundError(f"В папке нет изображений: {input_path}")
    rows: list[dict[str, str | int | float]] = []
    skipped_images = 0
    missing_masks = 0
    for image_index, image_path in enumerate(sources, start=1):
        image = read_image(image_path)
        if image is None:
            print(f"Пропущено повреждённое изображение: {image_path}")
            skipped_images += 1
            continue
        source_relative = image_path.relative_to(input_path).as_posix()
        card_id = get_card_id(image_path, input_path)
        stem = f"{image_index:04d}_{safe_stem(image_path.stem)}"
        product_mask, mask_source = load_product_mask(image_path, input_path, masks_path, image)
        if product_mask is None:
            missing_masks += 1

        def save_sample(sample_name: str, variant: str, sample_image: np.ndarray, sample_mask: np.ndarray | None, applied: dict[str, dict[str, Any]], order: list[str]) -> None:
            image_file = images_output / f"{sample_name}.jpg"
            save_jpeg(image_file, sample_image, jpeg_quality)
            mask_filename = ""
            if sample_mask is not None:
                mask_file = masks_output / f"{sample_name}.png"
                save_mask(mask_file, sample_mask)
                mask_filename = mask_file.relative_to(output_path).as_posix()
            rows.append(create_label_row(image_file.relative_to(output_path).as_posix(), source_relative, card_id, variant, mask_source, mask_filename, applied, order))

        save_sample(f"{stem}_clean", "clean", image, product_mask, {}, [])
        available_defects = [name for name in DEFECT_NAMES if product_mask is not None or name not in MASK_REQUIRED_DEFECTS]
        for defect_name in available_defects:
            for sample_number in range(1, singles_per_defect + 1):
                result, transformed_mask, applied, order = apply_defects(image, product_mask, [defect_name])
                save_sample(f"{stem}_single_{defect_name}_{sample_number:02d}", "single", result, transformed_mask, applied, order)
        for combination_number in range(1, combos_per_image + 1):
            maximum_count = min(maximum_combo_defects, len(available_defects))
            minimum_count = min(minimum_combo_defects, maximum_count)
            if maximum_count < 2:
                break
            defect_count = random.randint(minimum_count, maximum_count)
            selected_defects = random.sample(available_defects, defect_count)
            result, transformed_mask, applied, order = apply_defects(image, product_mask, selected_defects)
            save_sample(f"{stem}_combo_{combination_number:02d}", "combo", result, transformed_mask, applied, order)
    if not rows:
        raise RuntimeError("Не удалось создать ни одного изображения")
    fieldnames = [
        "filename",
        "source_image",
        "card_id",
        "variant",
        "is_clean",
        "defect_count",
        "mask_source",
        "mask_filename",
    ]
    for defect_name in DEFECT_NAMES:
        fieldnames.extend((f"{defect_name}_present", f"{defect_name}_intensity"))
    fieldnames.append("parameters_json")
    labels_file = output_path / "labels.csv"
    with labels_file.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    clean_count = sum(int(row["is_clean"]) for row in rows)
    single_count = sum(row["variant"] == "single" for row in rows)
    combo_count = sum(row["variant"] == "combo" for row in rows)
    print(f"Исходных изображений: {len(sources)}")
    print(f"Пропущено изображений: {skipped_images}")
    print(f"Изображений без маски: {missing_masks}")
    print(f"Создано clean: {clean_count}")
    print(f"Создано single: {single_count}")
    print(f"Создано combo: {combo_count}")
    print(f"Всего создано: {len(rows)}")
    print(f"Разметка: {labels_file}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генерация разнообразного синтетического датасета дефектов")
    parser.add_argument("--input_dir", required=True, help="Папка с исходными корректными изображениями")
    parser.add_argument("--output_dir", required=True, help="Папка для созданного датасета")
    parser.add_argument("--masks_dir", default=None, help="Необязательная папка с готовыми масками")
    parser.add_argument("--singles_per_defect", type=int, default=2, help="Количество одиночных примеров каждого дефекта")
    parser.add_argument("--combos_per_image", type=int, default=8, help="Количество комбинаций дефектов для одного исходника")
    parser.add_argument("--minimum_combo_defects", type=int, default=2, help="Минимальное количество дефектов в комбинации")
    parser.add_argument("--maximum_combo_defects", type=int, default=4, help="Максимальное количество дефектов в комбинации")
    parser.add_argument("--seed", type=int, default=42, help="Seed генератора")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="Качество сохранения JPEG")
    parser.add_argument("--overwrite", action="store_true", help="Удалить старый синтетический датасет и создать новый")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    generate_dataset(args.input_dir, args.output_dir, args.masks_dir, args.singles_per_defect, args.combos_per_image, args.minimum_combo_defects, args.maximum_combo_defects, args.seed, args.jpeg_quality, args.overwrite)

if __name__ == "__main__":
    main()