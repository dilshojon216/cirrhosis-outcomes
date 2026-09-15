"""Ma'lumotlarni yuklash, target'ni encode qilish, UCI original datani qo'shish.

Null strategiyasi (EDA xulosasi — ``notebooks/01_eda.ipynb``, 1-punkt):
    Klinik ustunlardagi (~43%) null'lar tasodifiy emas — ular blok holida
    birga tushadi (null-indikator korrelyatsiyasi 0.78-1.0), chunki asl
    Mayo Clinic tadqiqotida bir guruh bemorda to'liq klinik tekshiruv
    bo'lgan, boshqasida faqat asosiy kuzatuv ma'lumoti bo'lgan.

    Qaror: bu modulda hech qanday to'ldirish (mean/median) qilinmaydi —
    NaN qiymatlar xom holda qoldiriladi.
      - LightGBM/XGBoost/CatBoost null'larni o'zlari to'g'ri yo'naltiradi
        (missing-value routing) — median bilan to'ldirish esa soxta signal
        yaratadi va shu tabiiy afzallikni yo'qotadi.
      - "Bemor to'liq tekshiruvdan o'tganmi" belgisi (``is_full_labs``)
        alohida feature sifatida ``src/features.py`` da qo'shiladi (Faza 3).
      - Null'ga toqat qilmaydigan modellar (masalan LogisticRegression)
        uchun to'ldirish shu modelning o'z pipeline'ida (``SimpleImputer``)
        amalga oshiriladi — bu yerda emas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.utils import get_logger

logger = get_logger(__name__)

# Original UCI datasetdagi id ustuni raqamlari (1-418) train/test id'lari
# (0-24999) bilan kesishmasligi uchun shu qiymat qo'shiladi.
ORIGINAL_ID_OFFSET = 1_000_000


def get_label_maps(cfg: Config) -> tuple[dict[str, int], dict[int, str]]:
    """``cfg.data.classes`` tartibiga asoslangan Status <-> label mappinglari.

    Tartib ``config.yaml`` dagi ``data.classes`` (``[C, CL, D]``) bilan bir xil —
    bu submission ustunlari (``Status_C, Status_CL, Status_D``) tartibiga mos keladi.
    """
    status_to_label = {status: i for i, status in enumerate(cfg.data.classes)}
    label_to_status = {i: status for status, i in status_to_label.items()}
    return status_to_label, label_to_status


def load_raw(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``train.csv`` va ``test.csv`` ni o'qiydi, ``id`` ustunini index qiladi."""
    train = pd.read_csv(cfg.paths.train).set_index(cfg.data.id_col)
    test = pd.read_csv(cfg.paths.test).set_index(cfg.data.id_col)
    logger.info("load_raw: train=%s test=%s", train.shape, test.shape)
    return train, test


def encode_target(df: pd.DataFrame, cfg: Config, drop_invalid: bool = True) -> pd.DataFrame:
    """``Status`` ustunini ``C/CL/D -> 0/1/2`` ga o'giradi.

    EDA'da (``01_eda.ipynb``, 2-bo'lim) train'da ``cfg.data.classes`` ga
    kirmaydigan 1 ta anomal qiymat (``"Y"``) topilgan edi. ``drop_invalid=True``
    (standart) bo'lganda bunday qatorlar log bilan ogohlantirilib tashlab
    yuboriladi; ``False`` bo'lsa xatolik ko'tariladi.
    """
    target = cfg.data.target
    status_to_label, _ = get_label_maps(cfg)

    valid_mask = df[target].isin(cfg.data.classes)
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        bad_values = df.loc[~valid_mask, target].unique().tolist()
        if drop_invalid:
            logger.warning(
                "encode_target: %d qator kutilmagan Status qiymati bilan tashlandi: %s",
                n_invalid, bad_values,
            )
            df = df.loc[valid_mask]
        else:
            raise ValueError(f"Kutilmagan Status qiymatlari topildi: {bad_values}")

    df = df.copy()
    df[target] = df[target].map(status_to_label).astype("int8")
    return df


def decode_target(labels: pd.Series, cfg: Config) -> pd.Series:
    """``0/1/2 -> C/CL/D``. ``encode_target`` ning teskarisi."""
    _, label_to_status = get_label_maps(cfg)
    return labels.map(label_to_status)


def load_original(cfg: Config) -> pd.DataFrame:
    """Asl UCI ``cirrhosis.csv`` ni musobaqa ustun nomlariga moslashtiradi.

    Ustun nomlari va birliklari (``Age``, ``N_Days`` kunlarda) allaqachon
    musobaqa dataseti bilan bir xil — faqat ``ID -> id`` nomlanadi.
    """
    original = pd.read_csv(cfg.paths.original)
    original = original.rename(columns={"ID": cfg.data.id_col})
    original = original.set_index(cfg.data.id_col)

    expected_cols = set(cfg.data.numeric_cols) | set(cfg.data.categorical_cols) | {cfg.data.target}
    missing = expected_cols - set(original.columns)
    if missing:
        raise ValueError(f"UCI datasetda kutilgan ustunlar yo'q: {sorted(missing)}")

    logger.info("load_original: %s (UCI, %d null Stage)", original.shape, original["Stage"].isna().sum())
    return original


def merge_original(
    train: pd.DataFrame,
    original: pd.DataFrame,
    id_offset: int = ORIGINAL_ID_OFFSET,
) -> pd.DataFrame:
    """Original UCI datani train'ga qo'shadi, ``is_original`` flag bilan.

    Faqat train'ga qo'shiladi — test/validatsiya original data bilan
    ifloslanmasligi kerak (reja hujjati, "Qoidalar" 3-band). CV bo'lish
    vaqtida original qatorlar har doim train fold'ga tushishi uchun
    ``train.py`` da fold split ``is_original == 0`` qatorlarga qarab
    qilinadi, keyin original qatorlar train fold'ga qo'shib qo'yiladi.
    """
    train = train.copy()
    original = original.copy()

    train["is_original"] = 0
    original["is_original"] = 1
    original.index = original.index + id_offset

    overlap = train.index.intersection(original.index)
    if len(overlap):
        raise ValueError(f"id kesishmasi topildi (offset yetarli emas): {list(overlap)[:5]}...")

    combined = pd.concat([train, original], axis=0)
    logger.info(
        "merge_original: train=%d + original=%d -> combined=%d",
        len(train), len(original), len(combined),
    )
    return combined


def save_processed(df: pd.DataFrame, cfg: Config, name: str) -> Path:
    """DataFrame'ni ``data/processed/<name>.parquet`` ga saqlaydi."""
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.paths.processed_dir / f"{name}.parquet"
    df.to_parquet(out_path)
    logger.info("save_processed: %s (%s)", out_path, df.shape)
    return out_path


def load_processed(cfg: Config, name: str) -> pd.DataFrame:
    """``data/processed/<name>.parquet`` ni o'qiydi."""
    return pd.read_parquet(cfg.paths.processed_dir / f"{name}.parquet")
