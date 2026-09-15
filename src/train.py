"""CV fold generatsiyasi, baseline modellar, OOF ehtimolliklarni saqlash.

CV strategiyasi (reja hujjati, "Qoidalar" 3-band):
    ``StratifiedKFold`` FAQAT musobaqa train qatorlarida (``is_original == 0``)
    qurilib, validatsiya qatorlari FAQAT shulardan tanlanadi. Original UCI
    qatorlari (``is_original == 1``) hech qachon validatsiyaga tushmaydi —
    mavjud bo'lganda ular faqat train tomonga qo'shiladi (``fold == -1``).

Baseline'lar uchun feature'lar:
    Faqat raqamli ustunlar ishlatiladi (``get_numeric_feature_cols``) — raw
    kategorial ustunlar (``Drug``, ``Sex``, ...) chiqarib tashlanadi, chunki
    ular ``src/features.py`` da CatBoost uchun xom holida saqlab qo'yilgan
    (Faza 5). LogisticRegression uchun NaN'lar ``SimpleImputer`` bilan
    to'ldiriladi (o'z pipeline'i ichida — ``src/data.py`` dagi "to'ldirmaslik"
    qarori faqat xom datasetga tegishli); ``HistGradientBoostingClassifier``
    esa NaN'ni tabiiy qo'llab-quvvatlaydi, imputatsiya kerak emas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.data import encode_target, load_original, load_raw, merge_original
from src.features import FeatureEngineer
from src.utils import get_logger, seed_everything, timer

logger = get_logger(__name__)

# Faqat CatBoost'ga xom holida beriladigan kategorial ustunlar (Faza 5) —
# boshqa modellar uchun bular o'rniga features.py dagi *_enc/*_ord ustunlari ishlatiladi.
RAW_CATEGORICAL_COLS = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
META_COLS = {"fold", "is_original"}


def get_numeric_feature_cols(df: pd.DataFrame, cfg: Config) -> list[str]:
    """Raw kategorial va meta ustunlarni chiqarib, faqat raqamli feature'larni qaytaradi."""
    exclude = META_COLS | {cfg.data.target} | set(RAW_CATEGORICAL_COLS)
    return [c for c in df.columns if c not in exclude]


def make_folds(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Har bir qatorga fold raqami (0..n_folds-1) beradi.

    Original qatorlar (``is_original == 1``) hech qachon validatsiyaga
    tushmaydi — ularga ``-1`` beriladi.
    """
    folds = np.full(len(df), -1, dtype="int8")
    is_original = df["is_original"].to_numpy()
    comp_positions = np.flatnonzero(is_original == 0)
    y_comp = df[cfg.data.target].to_numpy()[comp_positions]

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (_, val_pos) in enumerate(skf.split(comp_positions, y_comp)):
        folds[comp_positions[val_pos]] = fold
    return folds


def prepare_train_test(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train/test'ni yuklaydi, target'ni encode qiladi, (ixtiyoriy) original
    datani qo'shadi, feature'larni quradi va ``fold`` ustunini belgilaydi."""
    train, test = load_raw(cfg)
    train = encode_target(train, cfg)

    if cfg.data.use_original:
        original = load_original(cfg)
        train = merge_original(train, original)
    else:
        train["is_original"] = 0

    fe = FeatureEngineer(cfg)
    train = fe.fit_transform(train)
    test = fe.transform(test)

    train["fold"] = make_folds(train, cfg)
    return train, test


def _align_proba_columns(proba: np.ndarray, classes_: np.ndarray, n_classes: int) -> np.ndarray:
    """``model.classes_`` ba'zi fold'da to'liq bo'lmasa (kam ehtimol), ustunlarni to'g'ri joyga qo'yadi."""
    if len(classes_) == n_classes and np.array_equal(classes_, np.arange(n_classes)):
        return proba
    aligned = np.zeros((proba.shape[0], n_classes))
    for i, c in enumerate(classes_):
        aligned[:, int(c)] = proba[:, i]
    return aligned


@dataclass
class CVResult:
    name: str
    oof_proba: np.ndarray
    cv_log_loss: float
    fold_log_losses: list[float]


def run_cv(model_builder, train: pd.DataFrame, feature_cols: list[str], cfg: Config, name: str) -> CVResult:
    """``model_builder()`` bilan StratifiedKFold CV yuritadi, OOF ehtimolliklarni qaytaradi.

    ``model_builder`` har chaqirilganda YANGI, fit qilinmagan model
    qaytaradigan callable bo'lishi kerak. Baholash faqat ``is_original == 0``
    (musobaqa) qatorlarda qilinadi.
    """
    target = cfg.data.target
    n_classes = len(cfg.data.classes)
    comp_mask = (train["is_original"] == 0).to_numpy()

    oof_proba = np.full((len(train), n_classes), np.nan)
    fold_losses = []

    for fold in range(cfg.n_folds):
        val_mask = (train["fold"] == fold).to_numpy()
        # original qatorlar (comp_mask=False) fold raqamidan qat'iy nazar doim train'ga kiradi
        fit_mask = (comp_mask & (train["fold"] != fold).to_numpy()) | (~comp_mask)

        X_fit, y_fit = train.loc[fit_mask, feature_cols], train.loc[fit_mask, target]
        X_val, y_val = train.loc[val_mask, feature_cols], train.loc[val_mask, target]

        model = model_builder()
        model.fit(X_fit, y_fit)
        proba = _align_proba_columns(model.predict_proba(X_val), model.classes_, n_classes)

        oof_proba[np.flatnonzero(val_mask)] = proba
        fold_loss = log_loss(y_val, proba, labels=list(range(n_classes)))
        fold_losses.append(fold_loss)
        logger.info("%s | fold %d: log_loss=%.5f (n_fit=%d, n_val=%d)", name, fold, fold_loss, fit_mask.sum(), val_mask.sum())

    overall_loss = log_loss(train.loc[comp_mask, target], oof_proba[comp_mask], labels=list(range(n_classes)))
    logger.info("%s | CV log_loss=%.5f (+/- %.5f)", name, overall_loss, float(np.std(fold_losses)))
    return CVResult(name=name, oof_proba=oof_proba, cv_log_loss=overall_loss, fold_log_losses=fold_losses)


def save_oof(result: CVResult, cfg: Config) -> Path:
    """OOF ehtimolliklarni ``outputs/oof/oof_<name>.npy`` ga saqlaydi."""
    cfg.paths.oof_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.oof_dir / f"oof_{result.name}.npy"
    np.save(path, result.oof_proba)
    logger.info("save_oof: %s", path)
    return path


def build_dummy_model() -> DummyClassifier:
    """Sinf chastotalari bilan bashorat qiluvchi pol (floor) modeli."""
    return DummyClassifier(strategy="prior")


def build_logreg_model(cfg: Config) -> Pipeline:
    params = cfg.models.get("logreg", {})
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=cfg.seed, **params)),
    ])


def build_hgb_model(cfg: Config) -> HistGradientBoostingClassifier:
    params = cfg.models.get("hgb", {})
    return HistGradientBoostingClassifier(random_state=cfg.seed, **params)


BASELINE_BUILDERS = {
    "dummy": lambda cfg: build_dummy_model(),
    "logreg": build_logreg_model,
    "hgb": build_hgb_model,
}


@timer
def run_baselines(cfg: Config) -> pd.DataFrame:
    """Barcha baseline modellarni (dummy, logreg, hgb) CV'da yuritadi va natijalar jadvalini qaytaradi."""
    seed_everything(cfg.seed)
    train, _ = prepare_train_test(cfg)
    feature_cols = get_numeric_feature_cols(train, cfg)
    logger.info("Baseline feature ustunlari (%d): %s", len(feature_cols), feature_cols)

    rows = []
    for name, builder in BASELINE_BUILDERS.items():
        result = run_cv(lambda b=builder: b(cfg), train, feature_cols, cfg, name)
        save_oof(result, cfg)
        rows.append({
            "model": name,
            "cv_log_loss": result.cv_log_loss,
            "cv_std": float(np.std(result.fold_log_losses)),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    results = run_baselines(cfg)
    print(results.to_string(index=False))
