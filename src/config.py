"""``config/config.yaml`` ni o'qib, tipini tekshirilgan dataclass'larga aylantiradi.

``models`` bo'limi qasddan oddiy ``dict`` sifatida qoldirilgan — har bir
kutubxona (LightGBM/XGBoost/CatBoost) o'z parametr to'plamiga ega, va ular
``**cfg.models["lgbm"]`` ko'rinishida to'g'ridan-to'g'ri modelga uzatiladi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    oof_dir: Path
    submissions_dir: Path
    figures_dir: Path
    train: Path
    test: Path
    sample_submission: Path
    original: Path


@dataclass(frozen=True)
class DataConfig:
    id_col: str
    target: str
    classes: list[str]
    use_original: bool
    numeric_cols: list[str]
    categorical_cols: list[str]
    log_cols: list[str]


@dataclass(frozen=True)
class Config:
    seed: int
    n_folds: int
    paths: PathsConfig
    data: DataConfig
    models: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """``config.yaml`` ni o'qiydi. Nisbiy yo'llar repo ildiziga nisbatan hisoblanadi."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    repo_root = path.resolve().parent.parent
    paths_raw = raw["paths"]
    paths = PathsConfig(**{k: repo_root / v for k, v in paths_raw.items()})

    data = DataConfig(**raw["data"])

    return Config(
        seed=raw["seed"],
        n_folds=raw["n_folds"],
        paths=paths,
        data=data,
        models=raw.get("models", {}),
    )
