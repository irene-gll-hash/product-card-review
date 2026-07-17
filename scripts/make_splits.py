"""Разделение датасета на train, validation и test по card_id."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def make_splits(labels_file: str | Path, output_file: str | Path, train_ratio: float = 0.70, validation_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42, overwrite: bool = False) -> None:
    labels_path = Path(labels_file).resolve()
    output_path = Path(output_file).resolve()

    if not labels_path.is_file():
        raise FileNotFoundError(f"Файл разметки не найден: {labels_path}")
    if output_path == labels_path:
        raise ValueError("output_file не должен совпадать с labels_file")
    if output_path.is_dir():
        raise IsADirectoryError(f"Вместо файла указана папка: {output_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Файл уже существует: {output_path}. Используй --overwrite для замены")

    ratios_sum = train_ratio + validation_ratio + test_ratio
    if any(ratio <= 0 for ratio in (train_ratio, validation_ratio, test_ratio)):
        raise ValueError("Все доли выборок должны быть больше нуля")
    if abs(ratios_sum - 1.0) > 1e-9:
        raise ValueError(f"Сумма долей должна быть равна 1.0, получено {ratios_sum}")

    labels = pd.read_csv(labels_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if labels.empty:
        raise ValueError("labels.csv не содержит строк")
    if "card_id" not in labels.columns:
        raise ValueError("В labels.csv отсутствует колонка card_id")

    labels["card_id"] = labels["card_id"].str.strip()
    if labels["card_id"].eq("").any():
        raise ValueError("В labels.csv присутствуют пустые card_id")

    card_ids = sorted(labels["card_id"].unique())
    if len(card_ids) < 4:
        raise ValueError("Для разделения на train, validation и test нужно минимум четыре уникальных card_id")

    temporary_ratio = validation_ratio + test_ratio
    try:
        train_ids, temporary_ids = train_test_split(card_ids, test_size=temporary_ratio, random_state=seed, shuffle=True)
        relative_test_ratio = test_ratio / temporary_ratio
        validation_ids, test_ids = train_test_split(temporary_ids, test_size=relative_test_ratio, random_state=seed, shuffle=True)
    except ValueError as error:
        raise ValueError("Не удалось разделить card_id с указанными долями") from error

    split_by_card = {card_id: "train" for card_id in train_ids}
    split_by_card.update({card_id: "validation" for card_id in validation_ids})
    split_by_card.update({card_id: "test" for card_id in test_ids})

    train_set = set(train_ids)
    validation_set = set(validation_ids)
    test_set = set(test_ids)

    if train_set & validation_set or train_set & test_set or validation_set & test_set:
        raise RuntimeError("Один card_id попал в несколько выборок")
    if len(split_by_card) != len(card_ids):
        raise RuntimeError("Не все card_id получили выборку")

    splits = pd.DataFrame({"card_id": card_ids})
    splits["split"] = splits["card_id"].map(split_by_card)

    labels_with_splits = labels.copy()
    labels_with_splits["split"] = labels_with_splits["card_id"].map(split_by_card)
    if labels_with_splits["split"].isna().any():
        raise RuntimeError("Некоторые изображения не получили выборку")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Всего карточек: {len(card_ids)}")
    print(f"Всего изображений: {len(labels)}")

    for split_name in ("train", "validation", "test"):
        cards_count = int((splits["split"] == split_name).sum())
        images_count = int((labels_with_splits["split"] == split_name).sum())
        print(f"{split_name}: {cards_count} карточек, {images_count} изображений")

    print(f"Разделение сохранено: {output_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Разделение датасета по card_id")
    parser.add_argument("--labels_file", default="data/synthetic/labels.csv", help="Путь к labels.csv")
    parser.add_argument("--output_file", default="data/splits.csv", help="Путь для сохранения splits.csv")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Доля train")
    parser.add_argument("--validation_ratio", type=float, default=0.15, help="Доля validation")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Доля test")
    parser.add_argument("--seed", type=int, default=42, help="Seed разделения")
    parser.add_argument("--overwrite", action="store_true", help="Перезаписать существующий splits.csv")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    make_splits(args.labels_file, args.output_file, args.train_ratio, args.validation_ratio, args.test_ratio, args.seed, args.overwrite)

if __name__ == "__main__":
    main()