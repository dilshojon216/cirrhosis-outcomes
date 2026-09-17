"""Kichik sintetik sample'da to'liq pipeline'ni oxirigacha ishga tushiradi.

Haqiqiy Kaggle/UCI ma'lumotisiz (CI internetga bog'liq bo'lmasligi uchun) —
``merge_original -> encode_target -> FeatureEngineer -> fold -> run_cv``
zanjirini bir butun sifatida tekshiradi. Tezkor bo'lishi uchun LightGBM
o'rniga oddiy sklearn pipeline ishlatiladi (``run_cv`` uchun model
kutubxonasi ahamiyatsiz — u har qanday ``fit``/``predict_proba`` bilan ishlaydi).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import encode_target, merge_original
from src.features import FeatureEngineer
from src.train import get_numeric_feature_cols, make_folds, run_cv


def _make_synthetic(n_per_class: int, seed: int) -> pd.DataFrame:
    """Real sxemaga o'xshash, ``Status`` bo'yicha muvozanatli sintetik dataset."""
    rng = np.random.default_rng(seed)
    classes = ["C", "CL", "D"] * n_per_class
    n = len(classes)
    return pd.DataFrame(
        {
            "N_Days": rng.integers(100, 5000, n),
            "Age": rng.integers(9000, 28000, n),
            "Bilirubin": rng.uniform(0.3, 10, n),
            "Cholesterol": rng.choice([np.nan, *rng.uniform(150, 400, 5)], n),
            "Albumin": rng.uniform(2.5, 4.5, n),
            "Copper": rng.choice([np.nan, *rng.uniform(10, 200, 5)], n),
            "Alk_Phos": rng.choice([np.nan, *rng.uniform(500, 3000, 5)], n),
            "SGOT": rng.choice([np.nan, *rng.uniform(50, 300, 5)], n),
            "Tryglicerides": rng.uniform(50, 200, n),
            "Platelets": rng.uniform(100, 400, n),
            "Prothrombin": rng.uniform(9, 14, n),
            "Stage": rng.integers(1, 5, n),
            "Drug": rng.choice(["Placebo", "D-penicillamine", None], n),
            "Sex": rng.choice(["F", "M"], n),
            "Ascites": rng.choice(["N", "Y", None], n),
            "Hepatomegaly": rng.choice(["N", "Y", None], n),
            "Spiders": rng.choice(["N", "Y", None], n),
            "Edema": rng.choice(["N", "S", "Y"], n),
            "Status": classes,
        },
        index=pd.Index(range(n), name="id"),
    )


def test_full_pipeline_smoke(cfg):
    """merge_original -> encode_target -> features -> fold -> CV — hech qanday xatosiz o'tishi kerak."""
    train_raw = _make_synthetic(n_per_class=20, seed=42)  # 60 qator
    original_raw = _make_synthetic(n_per_class=5, seed=7)  # 15 qator, "original" simulyatsiyasi

    train = merge_original(train_raw, original_raw)
    train = encode_target(train, cfg)  # merge'dan KEYIN — src/train.py dagi bug fix'ga mos tartib

    fe = FeatureEngineer(cfg, groups=("categorical",))
    train = fe.fit_transform(train)

    train["fold"] = make_folds(train, cfg)
    feature_cols = get_numeric_feature_cols(train, cfg)

    def build_model():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=cfg.seed)),
        ])

    result = run_cv(build_model, train, feature_cols, cfg, "smoke", log_progress=False)

    assert result.oof_proba.shape == (len(train), len(cfg.data.classes))
    assert not np.isnan(result.cv_log_loss)
    assert 0 < result.cv_log_loss < 5  # aqlga muvofiq oraliq (NaN/inf emas)

    # Original qatorlar (is_original==1) hech qachon OOF olmasligi kerak
    is_original = train["is_original"].to_numpy()
    assert np.isnan(result.oof_proba[is_original == 1]).all()
    assert not np.isnan(result.oof_proba[is_original == 0]).any()


def test_make_folds_never_validates_original_rows(cfg):
    train_raw = _make_synthetic(n_per_class=10, seed=1)
    original_raw = _make_synthetic(n_per_class=3, seed=2)
    train = merge_original(train_raw, original_raw)
    train = encode_target(train, cfg)

    folds = make_folds(train, cfg)
    is_original = train["is_original"].to_numpy()

    assert (folds[is_original == 1] == -1).all()
    assert (folds[is_original == 0] != -1).all()
    assert set(np.unique(folds[is_original == 0])) == set(range(cfg.n_folds))


def test_encode_target_drops_anomalous_status(cfg):
    df = pd.DataFrame({"Status": ["C", "CL", "D", "Y"]}, index=pd.Index(range(4), name="id"))
    out = encode_target(df, cfg)
    assert len(out) == 3
    assert set(out["Status"].unique()) == {0, 1, 2}


def test_encode_target_raises_when_drop_disabled(cfg):
    df = pd.DataFrame({"Status": ["C", "Y"]}, index=pd.Index(range(2), name="id"))
    with pytest.raises(ValueError):
        encode_target(df, cfg, drop_invalid=False)
