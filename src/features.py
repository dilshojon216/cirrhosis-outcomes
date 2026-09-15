"""Feature engineering: log1p, kategorial encoding, nisbat va klinik feature'lar.

Dizayn qarorlari:

- **Xom ustunlar hech qachon o'chirilmaydi/almashtirilmaydi** — har bir
  transformatsiya yangi ustun sifatida qo'shiladi (masalan ``Bilirubin`` ->
  ``Bilirubin_log1p``). Sabab: CatBoost kategorial ustunlarni xom holida
  (string) kutadi (Faza 5), LightGBM/XGBoost/LogReg esa encode qilinganini.
  Xom va encode qilingan versiya bir vaqtda mavjud bo'lsa, har bir model
  o'ziga keragini tanlab oladi.
- **Har bir funksiya o'z-o'zicha ishlaydi** — bir-biriga bog'liq emas (masalan
  ``add_ratio_features`` yosh/yil hisobini o'z ichida qayta hisoblaydi, uni
  ``add_age_years`` dan olmaydi). Shu sababli ``build_features`` da istalgan
  guruh kombinatsiyasi yoqib/o'chirib CV'da sinab ko'rilishi mumkin (reja
  hujjati, Faza 3 oxirgi bandi).
- **NaN qiymatlar tarqaladi, to'ldirilmaydi** — ``src/data.py`` dagi null
  strategiyasiga mos (EDA #1: blok holidagi MNAR null'lar).
- **fit/transform ajratilgan** (``FeatureEngineer``) — hozircha barcha
  transformatsiyalar stateless (train statistikasiga tayanmaydi), lekin
  interfeys shunday qurilganki, kelajakda train'dan o'rganiladigan encoder
  (masalan target encoding) qo'shilganda uning statistikasi FAQAT
  ``fit(train_df)`` da hisoblanadi — ``transform()`` esa test'ga ham xavfsiz
  qo'llanadi (leakage yo'q).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config

# N/Y qiymatli kategoriyalar uchun ikkilik (0/1) mapping.
# Sex va Drug ham 2 qiymatli — shu jadvalda birga.
BINARY_MAPS: dict[str, dict[str, int]] = {
    "Ascites": {"N": 0, "Y": 1},
    "Hepatomegaly": {"N": 0, "Y": 1},
    "Spiders": {"N": 0, "Y": 1},
    "Sex": {"F": 0, "M": 1},
    "Drug": {"Placebo": 0, "D-penicillamine": 1},
}

# Edema 3 qiymatli va tabiiy tartibga ega (og'irlashib boradi) -> ordinal.
EDEMA_MAP: dict[str, int] = {"N": 0, "S": 1, "Y": 2}

# EDA'da (01_eda.ipynb, #1) topilgan blok holida birga null bo'ladigan
# klinik ustunlar. Sinf chegarasi qat'iy emas (sintetik shovqin bor) —
# shuning uchun qattiq bool o'rniga har bir qatordagi null SONI hisoblanadi.
BLOCK_MISSING_COLS: list[str] = [
    "Drug", "Ascites", "Hepatomegaly", "Spiders",
    "Cholesterol", "Copper", "Alk_Phos", "SGOT", "Tryglicerides",
]

FEATURE_GROUPS: tuple[str, ...] = (
    "log", "age_years", "categorical", "ratios", "clinical_score", "missing_indicator",
)


def add_log_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """``cfg.data.log_cols`` uchun ``log1p`` ustunlari qo'shadi (o'ngga qiya taqsimotlar)."""
    df = df.copy()
    for col in cfg.data.log_cols:
        df[f"{col}_log1p"] = np.log1p(df[col])
    return df


def add_age_years(df: pd.DataFrame) -> pd.DataFrame:
    """``Age`` (kunlarda) dan o'qilishi osonroq ``Age_Years`` qo'shadi."""
    df = df.copy()
    df["Age_Years"] = df["Age"] / 365.25
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Binary (`_enc`) va ordinal (`Edema_ord`) versiyalarni qo'shadi.

    ``Stage`` allaqachon 1-4 ordinal raqam sifatida kelgani uchun qayta
    encode qilinmaydi.
    """
    df = df.copy()
    for col, mapping in BINARY_MAPS.items():
        df[f"{col}_enc"] = df[col].map(mapping).astype("float32")
    df["Edema_ord"] = df["Edema"].map(EDEMA_MAP).astype("float32")
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Klinik nisbat feature'lar: ``Bilirubin/Albumin``, ``SGOT/Alk_Phos``, ``Copper/Age_years``."""
    df = df.copy()
    age_years = df["Age"] / 365.25
    df["Bilirubin_Albumin_ratio"] = df["Bilirubin"] / df["Albumin"]
    df["SGOT_AlkPhos_ratio"] = df["SGOT"] / df["Alk_Phos"]
    df["Copper_per_AgeYear"] = df["Copper"] / age_years
    return df


def add_clinical_score(df: pd.DataFrame) -> pd.DataFrame:
    """``Ascites + Hepatomegaly + Spiders + Edema`` yig'indisi (0-5 oralig'ida).

    To'rttala ustun ham bir xil blok-missing guruhga tegishli (EDA #1) —
    shu sabab ular birga null bo'lgan qatorlarda ``ClinicalScore`` ham NaN
    bo'ladi. Bu qasddan qilingan — ``n_missing_clinical``/``is_full_labs``
    bilan birga model uchun izchil signal beradi.
    """
    df = df.copy()
    ascites = df["Ascites"].map(BINARY_MAPS["Ascites"])
    hepatomegaly = df["Hepatomegaly"].map(BINARY_MAPS["Hepatomegaly"])
    spiders = df["Spiders"].map(BINARY_MAPS["Spiders"])
    edema_ord = df["Edema"].map(EDEMA_MAP)
    df["ClinicalScore"] = ascites + hepatomegaly + spiders + edema_ord
    return df


def add_missing_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """``n_missing_clinical`` (0-9) va ``is_full_labs`` (n_missing==0) qo'shadi.

    EDA #1: bu ustunlar bir guruh bemorlarda (to'liq tekshiruv o'tmagan)
    birga yo'qoladi. Aniq chegara yo'q (sintetik shovqin bor) — shuning
    uchun qattiq flag o'rniga null SONI ustuvor signal sifatida beriladi.
    """
    df = df.copy()
    block_cols = [c for c in BLOCK_MISSING_COLS if c in df.columns]
    n_missing = df[block_cols].isnull().sum(axis=1)
    df["n_missing_clinical"] = n_missing.astype("int16")
    df["is_full_labs"] = (n_missing == 0).astype("int8")
    return df


def build_features(
    df: pd.DataFrame,
    cfg: Config,
    groups: tuple[str, ...] = FEATURE_GROUPS,
) -> pd.DataFrame:
    """Tanlangan feature guruhlarini ketma-ket qo'llaydi.

    ``groups`` orqali har bir guruhni alohida yoqib/o'chirib CV'da sinash
    mumkin (masalan ``groups=("log",)`` faqat log feature'larni qo'shadi).
    """
    groups = set(groups)
    unknown = groups - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Noma'lum feature guruh(lar)i: {unknown}")

    df = df.copy()
    if "log" in groups:
        df = add_log_features(df, cfg)
    if "age_years" in groups:
        df = add_age_years(df)
    if "categorical" in groups:
        df = encode_categoricals(df)
    if "ratios" in groups:
        df = add_ratio_features(df)
    if "clinical_score" in groups:
        df = add_clinical_score(df)
    if "missing_indicator" in groups:
        df = add_missing_indicator(df)
    return df


@dataclass
class FeatureEngineer:
    """sklearn'ga o'xshash fit/transform interfeysi — leakage'ga qarshi disiplina.

    Hozirgi transformatsiyalar stateless, shuning uchun ``fit`` faqat
    validatsiya qiladi. Kelajakda train statistikasiga tayanadigan encoder
    qo'shilsa (masalan target encoding), uning holati shu yerda, faqat
    ``fit(train_df)`` ichida hisoblanishi kerak — ``transform()`` test'ga
    xavfsiz qo'llanadigan holga kelguncha.
    """

    cfg: Config
    groups: tuple[str, ...] = FEATURE_GROUPS
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, train_df: pd.DataFrame) -> FeatureEngineer:
        missing_log_cols = set(self.cfg.data.log_cols) - set(train_df.columns)
        if missing_log_cols:
            raise ValueError(f"log_cols train'da topilmadi: {missing_log_cols}")
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureEngineer.transform() dan oldin fit() chaqirilishi kerak")
        return build_features(df, self.cfg, groups=self.groups)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)
