"""pytest fixture'lari: testlar uchun sintetik ``Config`` va ``DataFrame``.

Haqiqiy Kaggle/UCI fayllarga bog'liq emas — CI muhitida internetsiz ham
ishlaydi. ``sample_df`` real sxemaga o'xshab qurilgan (bo'sh qiymatlar va
Faza 1'da topilgan anomal ``Status`` qiymati ("Y") ham qasddan qo'shilgan).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config, DataConfig, PathsConfig

NUMERIC_COLS = [
    "N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper",
    "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin",
]
CATEGORICAL_COLS = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"]
LOG_COLS = ["Bilirubin", "Copper", "Alk_Phos", "SGOT", "Cholesterol"]
CLASSES = ["C", "CL", "D"]


@pytest.fixture
def cfg(tmp_path) -> Config:
    """Haqiqiy fayllarga tegmaydigan, ``tmp_path`` asosidagi Config."""
    paths = PathsConfig(
        raw_dir=tmp_path / "data/raw",
        processed_dir=tmp_path / "data/processed",
        models_dir=tmp_path / "models",
        oof_dir=tmp_path / "outputs/oof",
        submissions_dir=tmp_path / "outputs/submissions",
        figures_dir=tmp_path / "outputs/figures",
        train=tmp_path / "data/raw/train.csv",
        test=tmp_path / "data/raw/test.csv",
        sample_submission=tmp_path / "data/raw/sample_submission.csv",
        original=tmp_path / "data/raw/cirrhosis.csv",
    )
    data = DataConfig(
        id_col="id",
        target="Status",
        classes=list(CLASSES),
        use_original=False,
        numeric_cols=list(NUMERIC_COLS),
        categorical_cols=list(CATEGORICAL_COLS),
        log_cols=list(LOG_COLS),
    )
    return Config(seed=42, n_folds=3, paths=paths, data=data, models={})


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """8 qatorli sintetik DataFrame — bo'sh qiymatlar va 1 ta anomal Status bilan."""
    return pd.DataFrame(
        {
            "N_Days": [100, 200, 300, 400, 500, 600, 700, 800],
            "Age": [10000, 12000, 15000, 20000, 25000, 30000, 18000, 22000],
            "Bilirubin": [0.5, 1.0, 2.0, np.nan, 3.0, 5.0, 0.8, np.nan],
            "Cholesterol": [200, 220, np.nan, 250, 260, 300, 210, np.nan],
            "Albumin": [3.5, 3.6, 3.7, 3.2, 3.0, 2.8, 3.4, 3.1],
            "Copper": [50, 60, np.nan, 80, 90, 100, 55, np.nan],
            "Alk_Phos": [1000, 1100, np.nan, 1300, 1400, 1500, 1050, np.nan],
            "SGOT": [100, 110, np.nan, 130, 140, 150, 105, np.nan],
            "Tryglicerides": [100, 110, 120, np.nan, 140, 150, 115, 125],
            "Platelets": [200, 210, 220, 230, np.nan, 250, 215, 225],
            "Prothrombin": [10, 10.5, 11, 11.5, 12, 12.5, 10.2, 11.8],
            "Stage": [1, 2, 3, 4, 3, 2, 1, 4],
            "Drug": ["Placebo", "D-penicillamine", None, "Placebo", "D-penicillamine", None, "Placebo", "D-penicillamine"],
            "Sex": ["F", "M", "F", "M", "F", "M", "F", "M"],
            "Ascites": ["N", "Y", None, "N", "Y", None, "N", "Y"],
            "Hepatomegaly": ["N", "Y", None, "N", "Y", None, "N", "Y"],
            "Spiders": ["N", "Y", None, "N", "Y", None, "N", "Y"],
            "Edema": ["N", "S", "Y", "N", "S", "Y", "N", "S"],
            "Status": ["C", "CL", "D", "C", "D", "C", "D", "Y"],  # oxirgisi — anomaliya (EDA #1)
        },
        index=pd.Index(range(8), name="id"),
    )
