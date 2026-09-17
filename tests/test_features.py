"""``src/features.py`` dagi transformatsiya funksiyalarini tekshiradi.

Fixture'lar ``conftest.py`` da (``cfg``, ``sample_df``).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features import (
    FEATURE_GROUPS,
    FeatureEngineer,
    add_age_years,
    add_clinical_score,
    add_log_features,
    add_missing_indicator,
    add_ratio_features,
    build_features,
    encode_categoricals,
)


def test_add_log_features_preserves_raw_and_adds_log1p(cfg, sample_df):
    out = add_log_features(sample_df, cfg)
    for col in cfg.data.log_cols:
        np.testing.assert_allclose(out[f"{col}_log1p"], np.log1p(sample_df[col]), equal_nan=True)
    assert set(cfg.data.log_cols).issubset(out.columns)  # xom ustun o'chirilmagan


def test_add_age_years(sample_df):
    out = add_age_years(sample_df)
    np.testing.assert_allclose(out["Age_Years"], sample_df["Age"] / 365.25)


def test_encode_categoricals_binary_and_ordinal(sample_df):
    out = encode_categoricals(sample_df)
    assert out.loc[0, "Ascites_enc"] == 0  # "N"
    assert out.loc[1, "Ascites_enc"] == 1  # "Y"
    assert np.isnan(out.loc[2, "Ascites_enc"])  # bo'sh qiymat
    assert out.loc[0, "Edema_ord"] == 0  # "N"
    assert out.loc[1, "Edema_ord"] == 1  # "S"
    assert out.loc[2, "Edema_ord"] == 2  # "Y"
    # Xom kategorial ustunlar saqlanadi (CatBoost uchun, Faza 5)
    assert "Ascites" in out.columns and "Edema" in out.columns


def test_add_ratio_features_nan_propagates(sample_df):
    out = add_ratio_features(sample_df)
    assert np.isnan(out.loc[3, "Bilirubin_Albumin_ratio"])  # Bilirubin[3] NaN
    assert out.loc[0, "Bilirubin_Albumin_ratio"] == pytest.approx(0.5 / 3.5)


def test_add_clinical_score_sum_and_nan(sample_df):
    out = add_clinical_score(sample_df)
    assert out.loc[0, "ClinicalScore"] == 0  # N+N+N+N
    assert out.loc[1, "ClinicalScore"] == 4  # Y(1)+Y(1)+Y(1)+S(1)
    assert np.isnan(out.loc[2, "ClinicalScore"])  # Ascites/Hepatomegaly/Spiders bo'sh


def test_add_missing_indicator(sample_df):
    out = add_missing_indicator(sample_df)
    assert out.loc[0, "is_full_labs"] == 1
    assert out.loc[0, "n_missing_clinical"] == 0
    assert out.loc[2, "is_full_labs"] == 0
    assert out.loc[2, "n_missing_clinical"] > 0


@pytest.mark.parametrize("groups", [(g,) for g in FEATURE_GROUPS])
def test_build_features_groups_are_independent(cfg, sample_df, groups):
    """Har bir feature guruhi boshqalarga bog'liq bo'lmasdan mustaqil ishlashi kerak."""
    out = build_features(sample_df, cfg, groups=groups)
    assert len(out) == len(sample_df)
    assert set(out.columns) - set(sample_df.columns)  # kamida bitta yangi ustun


def test_build_features_unknown_group_raises(cfg, sample_df):
    with pytest.raises(ValueError):
        build_features(sample_df, cfg, groups=("nosozgroup",))


def test_build_features_all_groups(cfg, sample_df):
    out = build_features(sample_df, cfg, groups=FEATURE_GROUPS)
    expected_new = (
        {f"{c}_log1p" for c in cfg.data.log_cols}
        | {"Age_Years", "Ascites_enc", "Hepatomegaly_enc", "Spiders_enc", "Sex_enc", "Drug_enc", "Edema_ord"}
        | {"Bilirubin_Albumin_ratio", "SGOT_AlkPhos_ratio", "Copper_per_AgeYear"}
        | {"ClinicalScore", "n_missing_clinical", "is_full_labs"}
    )
    assert expected_new.issubset(out.columns)


def test_feature_engineer_requires_fit_before_transform(cfg, sample_df):
    fe = FeatureEngineer(cfg)
    with pytest.raises(RuntimeError):
        fe.transform(sample_df)


def test_feature_engineer_fit_transform_roundtrip(cfg, sample_df):
    fe = FeatureEngineer(cfg, groups=("categorical",))
    out = fe.fit_transform(sample_df)
    assert "Ascites_enc" in out.columns
    out_subset = fe.transform(sample_df.iloc[:3])  # fit qilingandan keyin boshqa data'ga ham ishlaydi
    assert "Ascites_enc" in out_subset.columns


def test_feature_engineer_missing_log_cols_raises(cfg, sample_df):
    fe = FeatureEngineer(cfg)
    bad_df = sample_df.drop(columns=["Bilirubin"])
    with pytest.raises(ValueError):
        fe.fit(bad_df)
