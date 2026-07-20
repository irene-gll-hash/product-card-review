# Product Card Review

Инструменты подготовки датасета для поиска дефектов на изображениях карточек товаров.
Пакет использует `src`-layout и устанавливается под именем `card_checker`.

## Быстрый запуск

Требуются Python 3.11–3.13 и [uv](https://docs.astral.sh/uv/).

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv sync --dev
```

Проверка импортов и тестов:

```powershell
uv run python -c "import card_checker; from card_checker.data.dataset import ProductCardDataset; print('src imports: OK')"
uv run pytest -p no:cacheprovider
```

## Подготовка изображений

Исходные корректные карточки размещаются в `data/raw/good_images`. Затем создаются
варианты с дефектами, разбиение по выборкам и проверяется итоговый датасет:

```powershell
uv run python scripts/generate_defects.py --input_dir data/raw/good_images --output_dir data/synthetic --singles_per_defect 1 --combos_per_image 2 --overwrite
uv run python scripts/make_splits.py --overwrite
uv run python scripts/validate_dataset.py
```

Сгенерированные изображения находятся в `data/synthetic/images`, маски — в
`data/synthetic/masks`, разметка — в `data/synthetic/labels.csv`, а разбиение — в
`data/splits.csv`. Бинарные данные и локальное разбиение исключены из Git.

## Основные импорты

```python
from card_checker.data.dataset import ProductCardDataset
from card_checker.data.loaders import build_dataloaders, build_datasets
from card_checker.data.transforms import build_train_transforms
```
