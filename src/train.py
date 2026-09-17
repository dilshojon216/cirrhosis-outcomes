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

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

# make_lgbm_fit_kwargs() qasddan eski (lekin keng qo'llab-quvvatlanadigan) `eval_set`
# parametrini ishlatadi — bu ogohlantirish shu qarorning kutilgan natijasi.
warnings.filterwarnings("ignore", message=r"The argument 'eval_set' is deprecated")

# Faqat CatBoost'ga xom holida beriladigan kategorial ustunlar (Faza 5) —
# boshqa modellar uchun bular o'rniga features.py dagi *_enc/*_ord ustunlari ishlatiladi.
RAW_CATEGORICAL_COLS = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
# CatBoost xom ustunlarni oladi, shuning uchun ularning encode qilingan
# nusxalari (features.py) CatBoost feature to'plamidan chiqarib tashlanadi.
ENCODED_DUPLICATE_COLS = ["Ascites_enc", "Hepatomegaly_enc", "Spiders_enc", "Sex_enc", "Drug_enc", "Edema_ord"]
META_COLS = {"fold", "is_original"}


def get_numeric_feature_cols(df: pd.DataFrame, cfg: Config) -> list[str]:
    """Raw kategorial va meta ustunlarni chiqarib, faqat raqamli feature'larni qaytaradi."""
    exclude = META_COLS | {cfg.data.target} | set(RAW_CATEGORICAL_COLS)
    return [c for c in df.columns if c not in exclude]


def get_catboost_feature_cols(df: pd.DataFrame, cfg: Config) -> list[str]:
    """Raqamli feature'lar + xom kategorial ustunlar (CatBoost ``cat_features`` uchun)."""
    exclude = META_COLS | {cfg.data.target} | set(ENCODED_DUPLICATE_COLS)
    return [c for c in df.columns if c not in exclude]


def prepare_catboost_frame(df: pd.DataFrame) -> pd.DataFrame:
    """CatBoost kategorial ustunlardagi ``NaN``ni qabul qilmaydi (xato beradi) —
    ``"missing"`` satr bilan to'ldiradi. Bu aslida foydali: ``src/data.py`` dagi
    "to'ldirmaslik" qarori faqat RAQAMLI ustunlarga tegishli edi — kategorial
    ustun uchun "missing" alohida kategoriya sifatida CatBoost'ning o'ziga
    signal beradi (``is_full_labs`` bilan bir xil ma'lumot, boshqa shaklda).
    """
    df = df.copy()
    df[RAW_CATEGORICAL_COLS] = df[RAW_CATEGORICAL_COLS].fillna("missing")
    return df


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


def prepare_train_test(
    cfg: Config, groups: tuple[str, ...] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train/test'ni yuklaydi, (ixtiyoriy) original datani qo'shadi, target'ni
    encode qiladi, feature'larni quradi va ``fold`` ustunini belgilaydi.

    ``groups`` — ``FeatureEngineer``ga uzatiladigan feature guruhlari
    (``src/features.py``, ``FEATURE_GROUPS``). ``None`` bo'lsa barcha guruh
    qo'llanadi (standart). Faza 5/6/7 da eng yaxshi natija bergan
    konfiguratsiya — ``groups=("categorical",)`` ("baza") + ``use_original=True`` —
    chunki qo'shimcha feature'lar (log1p, nisbat, ...) daraxt modellari uchun
    CV'ni yomonlashtirgan (``03_experiments.ipynb``, §1).

    MUHIM: ``encode_target()`` ``merge_original()`` dan KEYIN chaqiriladi.
    Aks holda ``train["Status"]`` (allaqachon ``int8``) va ``original["Status"]``
    (hali ``"C"/"CL"/"D"`` satr) turli tipda birlashib, ``concat`` natijasida
    ustun ``object`` (aralash int+str) bo'lib qoladi va ``StratifiedKFold``
    "Supported target types" xatosi bilan portlaydi.
    """
    train, test = load_raw(cfg)

    if cfg.data.use_original:
        original = load_original(cfg)
        train = merge_original(train, original)
    else:
        train["is_original"] = 0

    train = encode_target(train, cfg)

    fe = FeatureEngineer(cfg, groups=groups) if groups is not None else FeatureEngineer(cfg)
    train = fe.fit_transform(train)
    test = fe.transform(test)

    train["fold"] = make_folds(train, cfg)
    return train, test


def _align_proba_columns(proba: np.ndarray, classes_: np.ndarray, n_classes: int) -> np.ndarray:
    """``model.classes_`` ba'zi fold'da to'liq bo'lmasa (kam ehtimol), ustunlarni to'g'ri joyga qo'yadi.

    Qatorlarni ham 1.0 ga normalizatsiya qiladi — XGBoost'ning ``float32``
    softmax chiqishi ba'zan ``~1e-7`` chetga chiqadi, sklearn ``log_loss``
    esa bunga ogohlantirish beradi (natija o'zi to'g'ri, lekin shovqinli).
    """
    if not (len(classes_) == n_classes and np.array_equal(classes_, np.arange(n_classes))):
        aligned = np.zeros((proba.shape[0], n_classes))
        for i, c in enumerate(classes_):
            aligned[:, int(c)] = proba[:, i]
        proba = aligned
    return proba / proba.sum(axis=1, keepdims=True)


@dataclass
class CVResult:
    name: str
    oof_proba: np.ndarray
    cv_log_loss: float
    fold_log_losses: list[float]
    models: list[Any] | None = field(default=None, repr=False)


def run_cv(
    model_builder: Callable[[], Any],
    train: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
    name: str,
    fit_kwargs_fn: Callable[[pd.DataFrame, pd.Series], dict[str, Any]] | None = None,
    sample_weight_fn: Callable[[pd.Series], np.ndarray] | None = None,
    save_models: bool = False,
    log_progress: bool = True,
) -> CVResult:
    """``model_builder()`` bilan StratifiedKFold CV yuritadi, OOF ehtimolliklarni qaytaradi.

    ``model_builder`` har chaqirilganda YANGI, fit qilinmagan model
    qaytaradigan callable bo'lishi kerak. Baholash faqat ``is_original == 0``
    (musobaqa) qatorlarda qilinadi.

    ``fit_kwargs_fn(X_val, y_val) -> dict`` — LightGBM/XGBoost/CatBoost uchun
    fold ICHIDA early stopping qilish uchun ``eval_set`` (yoki ekvivalenti)
    beradi (reja hujjati, Faza 5: "Har biriga early_stopping fold ichida").
    Izoh: bu shu fold'ning validatsiya qismidan foydalanadi — Kaggle
    amaliyotida keng tarqalgan yondashuv, lekin qat'iy ma'noda "to'xtash
    nuqtasi" val'ni ozgina ko'rgan bo'ladi (nested CV emas).
    """
    target = cfg.data.target
    n_classes = len(cfg.data.classes)
    comp_mask = (train["is_original"] == 0).to_numpy()

    oof_proba = np.full((len(train), n_classes), np.nan)
    fold_losses = []
    fold_models = [] if save_models else None

    for fold in range(cfg.n_folds):
        val_mask = (train["fold"] == fold).to_numpy()
        # original qatorlar (comp_mask=False) fold raqamidan qat'iy nazar doim train'ga kiradi
        fit_mask = (comp_mask & (train["fold"] != fold).to_numpy()) | (~comp_mask)

        X_fit, y_fit = train.loc[fit_mask, feature_cols], train.loc[fit_mask, target]
        X_val, y_val = train.loc[val_mask, feature_cols], train.loc[val_mask, target]

        fit_kwargs = fit_kwargs_fn(X_val, y_val) if fit_kwargs_fn is not None else {}
        if sample_weight_fn is not None:
            fit_kwargs["sample_weight"] = sample_weight_fn(y_fit)

        model = model_builder()
        model.fit(X_fit, y_fit, **fit_kwargs)
        raw_proba = model.predict_proba(X_val).astype("float64")  # float32 (XGBoost) rounding shovqinini kamaytiradi
        proba = _align_proba_columns(raw_proba, model.classes_, n_classes)

        oof_proba[np.flatnonzero(val_mask)] = proba
        fold_loss = log_loss(y_val, proba, labels=list(range(n_classes)))
        fold_losses.append(fold_loss)
        if save_models:
            fold_models.append(model)
        if log_progress:
            logger.info("%s | fold %d: log_loss=%.5f (n_fit=%d, n_val=%d)", name, fold, fold_loss, fit_mask.sum(), val_mask.sum())

    overall_loss = log_loss(train.loc[comp_mask, target], oof_proba[comp_mask], labels=list(range(n_classes)))
    if log_progress:
        logger.info("%s | CV log_loss=%.5f (+/- %.5f)", name, overall_loss, float(np.std(fold_losses)))
    return CVResult(name=name, oof_proba=oof_proba, cv_log_loss=overall_loss, fold_log_losses=fold_losses, models=fold_models)


def save_oof(result: CVResult, cfg: Config) -> Path:
    """OOF ehtimolliklarni ``outputs/oof/oof_<name>.npy`` ga saqlaydi."""
    cfg.paths.oof_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.oof_dir / f"oof_{result.name}.npy"
    np.save(path, result.oof_proba)
    logger.info("save_oof: %s", path)
    return path


def save_fold_models(models: list[Any], cfg: Config, name: str) -> list[Path]:
    """Har bir fold modelini ``models/<name>_fold<i>.pkl`` ga saqlaydi (``joblib``).

    ``src/predict.py`` shu fayllarni o'qib, qayta o'qitmasdan bashorat qiladi.
    """
    import joblib

    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, model in enumerate(models):
        path = cfg.paths.models_dir / f"{name}_fold{i}.pkl"
        joblib.dump(model, path)
        paths.append(path)
    logger.info("save_fold_models: %d model -> %s", len(paths), cfg.paths.models_dir)
    return paths


def load_fold_models(cfg: Config, name: str) -> list[Any]:
    """``save_fold_models()`` bilan saqlangan fold modellarini o'qiydi (fold tartibida).

    Xavfsizlik izohi: ``joblib.load`` pickle orqali o'qiydi (arbitrary code
    execution xavfi tashqi/ishonchsiz manbadan yuklaganda). Bu yerda fayllar
    faqat shu loyihaning o'zi (``save_fold_models``) tomonidan lokal
    ``models/`` papkasiga yozilgan — tashqi/ishonchsiz manbadan yuklanmaydi.
    """
    import joblib

    paths = sorted(
        cfg.paths.models_dir.glob(f"{name}_fold*.pkl"),
        key=lambda p: int(p.stem.rsplit("fold", 1)[1]),
    )
    if not paths:
        raise FileNotFoundError(
            f"'{name}' uchun saqlangan model topilmadi ({cfg.paths.models_dir}). "
            "Avval `python -m src.train` (yoki `make train`) ishga tushiring."
        )
    return [joblib.load(p) for p in paths]


# Faza 5/6/7 da tanlangan g'olib konfiguratsiya — production model shu bilan o'qitiladi.
FINAL_MODEL_NAME = "lgbm_final"
FINAL_FEATURE_GROUPS = ("categorical",)


def train_final_model(cfg: Config) -> CVResult:
    """Yakuniy production model: tuned LightGBM, "baza" feature to'plami
    (faqat categorical encoding) + original UCI data (``cfg.data.use_original``).

    Bu 03_experiments.ipynb/07_ensemble.ipynb da tanlangan g'olib konfiguratsiya:
    qo'shimcha feature'lar (log1p/nisbat/klinik ball) va ensemble/stacking
    sinovdan o'tib, hech biri yakka sozlangan LightGBM'dan ustun kelmagan edi.
    """
    seed_everything(cfg.seed)
    train, _ = prepare_train_test(cfg, groups=FINAL_FEATURE_GROUPS)
    feature_cols = get_numeric_feature_cols(train, cfg)
    early_stopping_rounds = cfg.models["lgbm"]["early_stopping_rounds"]

    result = run_cv(
        lambda: build_lgbm_model(cfg),
        train, feature_cols, cfg, FINAL_MODEL_NAME,
        fit_kwargs_fn=make_lgbm_fit_kwargs(early_stopping_rounds),
        save_models=True,
    )
    save_oof(result, cfg)
    save_fold_models(result.models, cfg, FINAL_MODEL_NAME)
    return result


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


def _pop_boosting_meta(params: dict[str, Any], n_estimators_key: str) -> tuple[dict[str, Any], int, int]:
    """``config.yaml`` dagi model bo'limidan iteratsiya soni va early-stopping
    parametrini ajratib oladi (ular constructor'ga boshqacha uzatiladi/uzatilmaydi)."""
    params = dict(params)
    n_estimators = params.pop(n_estimators_key)
    early_stopping_rounds = params.pop("early_stopping_rounds")
    return params, n_estimators, early_stopping_rounds


def build_lgbm_model(cfg: Config, **overrides: Any):
    import lightgbm as lgb

    params, n_estimators, _ = _pop_boosting_meta(cfg.models["lgbm"], "n_estimators")
    params.update(overrides)
    return lgb.LGBMClassifier(n_estimators=n_estimators, random_state=cfg.seed, **params)


def build_xgb_model(cfg: Config, **overrides: Any):
    import xgboost as xgb

    params, n_estimators, early_stopping_rounds = _pop_boosting_meta(cfg.models["xgb"], "n_estimators")
    params.update(overrides)
    return xgb.XGBClassifier(
        n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, random_state=cfg.seed, **params
    )


def build_catboost_model(cfg: Config, **overrides: Any):
    import catboost as cb

    params, iterations, early_stopping_rounds = _pop_boosting_meta(cfg.models["catboost"], "iterations")
    params.update(overrides)
    return cb.CatBoostClassifier(
        iterations=iterations, early_stopping_rounds=early_stopping_rounds, random_seed=cfg.seed, **params
    )


def make_lgbm_fit_kwargs(early_stopping_rounds: int) -> Callable[[pd.DataFrame, pd.Series], dict[str, Any]]:
    """LightGBM early-stopping uchun ``eval_set`` beradi.

    Qasddan yangi ``eval_X``/``eval_y`` (4.x) emas, eski ``eval_set`` ishlatiladi —
    u ozgina eskirgan (``LGBMDeprecationWarning`` beradi) lekin deyarli barcha
    LightGBM versiyalarida (jumladan Kaggle notebook muhitidagi ko'pincha eskiroq
    versiyalarda) ishlaydi; ``eval_X``/``eval_y`` esa u yerda ``TypeError`` bilan
    yiqiladi (haqiqiy hodisa — ``notebooks/04_kaggle_submission.ipynb`` da topilgan).
    """
    def _fn(X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        import lightgbm as lgb

        return {"eval_set": [(X_val, y_val)], "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)]}
    return _fn


def xgb_fit_kwargs(X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
    return {"eval_set": [(X_val, y_val)], "verbose": False}


def make_catboost_fit_kwargs(cat_features: list[str]) -> Callable[[pd.DataFrame, pd.Series], dict[str, Any]]:
    def _fn(X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        return {"eval_set": (X_val, y_val), "cat_features": cat_features}
    return _fn


def get_feature_importance(models: list[Any], feature_cols: list[str]) -> pd.DataFrame:
    """Fold modellaridan o'rtacha ``feature_importances_`` ni hisoblaydi
    (LightGBM/XGBoost/CatBoost sklearn API — barchasida shu atribut bor)."""
    importances = np.array([m.feature_importances_ for m in models], dtype=float)
    return pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": importances.mean(axis=0),
        "importance_std": importances.std(axis=0),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def tune_lgbm(
    train: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
    n_trials: int = 30,
    timeout: int | None = None,
):
    """Optuna bilan LightGBM giperparametrlarini CV log loss bo'yicha sozlaydi.

    Har bir trial to'liq ``cfg.n_folds`` marta CV yuritadi (fold ichida
    early stopping bilan) — sekinroq, lekin nested-CV shart emas, chunki
    baholash mezoni (CV log loss) allaqachon barcha fold'lar bo'yicha
    o'rtachalangan.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    early_stopping_rounds = cfg.models["lgbm"]["early_stopping_rounds"]
    fit_kwargs_fn = make_lgbm_fit_kwargs(early_stopping_rounds)
    n_classes = len(cfg.data.classes)

    def objective(trial: optuna.Trial) -> float:
        import lightgbm as lgb

        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            "verbose": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        def builder():
            return lgb.LGBMClassifier(n_estimators=5000, random_state=cfg.seed, **params)

        result = run_cv(
            builder, train, feature_cols, cfg, name=f"lgbm_trial{trial.number}",
            fit_kwargs_fn=fit_kwargs_fn, log_progress=False,
        )
        return result.cv_log_loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study


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
    # `python -m src.train` — production modelni o'qitadi va models/ ga saqlaydi
    # (`make train`). Faza 4 baseline'larini alohida ishga tushirish uchun:
    # `python -c "from src.config import load_config; from src.train import run_baselines; print(run_baselines(load_config()))"`
    from src.config import load_config

    cfg = load_config()
    result = train_final_model(cfg)
    print(f"Yakuniy model o'qitildi: CV log_loss = {result.cv_log_loss:.5f}")
    print(f"Fold modellari: {cfg.paths.models_dir}/{FINAL_MODEL_NAME}_fold*.pkl")
