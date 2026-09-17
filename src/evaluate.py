"""Model baholash: sinf bo'yicha log loss, confusion matrix, kalibratsiya, slice tahlili.

Barcha funksiyalar OOF (out-of-fold) ehtimolliklar ustida ishlaydi — ular
``src/train.py`` (``run_cv``/``save_oof``) tomonidan ``outputs/oof/`` ga
saqlangan. Faqat musobaqa (``is_original == 0``) qatorlar baholanadi.

OOF fayl shakli ikki xil bo'lishi mumkin (Faza 4 va Faza 5 turlicha
konfiguratsiyada saqlangani sabab):
    - ``(n_comp, 3)`` — faqat musobaqa qatorlari (``use_original=False`` paytida)
    - ``(n_comp + n_original, 3)`` — original qatorlar uchun ``NaN`` bilan
      (``use_original=True`` paytida, ``is_original==1`` qatorlar hech qachon
      OOF olmaydi)
``load_oof()`` shu ikkalasini ham avtomatik moslashtiradi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, log_loss

from src.config import Config
from src.utils import get_logger

logger = get_logger(__name__)


def load_oof(name: str, cfg: Config, comp_mask: np.ndarray) -> np.ndarray:
    """``outputs/oof/oof_<name>.npy`` ni o'qiydi va faqat musobaqa qatorlariga moslaydi.

    ``comp_mask`` — ``src.train.prepare_train_test`` dan olingan to'liq
    (``is_original`` qo'shilgan) train frame uchun bool mask.
    """
    path = cfg.paths.oof_dir / f"oof_{name}.npy"
    proba = np.load(path)
    n_comp = int(comp_mask.sum())

    if proba.shape[0] == n_comp:
        return proba
    if proba.shape[0] == len(comp_mask):
        return proba[comp_mask]
    raise ValueError(
        f"oof_{name}.npy shape {proba.shape} na comp_mask uzunligi ({len(comp_mask)}) "
        f"na musobaqa qatorlar soni ({n_comp}) bilan mos kelmadi"
    )


def overall_log_loss(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    """Musobaqa metrikasi bilan bir xil — barcha sinflar bo'yicha o'rtacha log loss."""
    return float(log_loss(y_true, proba, labels=list(range(n_classes))))


def per_class_log_loss(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3, eps: float = 1e-15) -> pd.Series:
    """Har bir sinf uchun ALOHIDA log loss: ``-mean(log(p[haqiqiy_sinf]))`` faqat shu
    sinfga tegishli qatorlarda. ``overall_log_loss`` dan farqli — bu qaysi sinfda
    model ko'proq "jazolanayotganini" ko'rsatadi (masalan kam uchraydigan ``CL``).
    """
    proba = np.clip(proba, eps, 1 - eps)
    out = {}
    for c in range(n_classes):
        mask = y_true == c
        out[c] = float(-np.log(proba[mask, c]).mean()) if mask.any() else float("nan")
    return pd.Series(out, name="log_loss")


def confusion_matrix_df(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Argmax bashorat asosida confusion matrix (qatorlar=haqiqiy, ustunlar=bashorat)."""
    y_pred = proba.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    df = pd.DataFrame(cm, index=[f"haqiqiy_{c}" for c in classes], columns=[f"bashorat_{c}" for c in classes])
    return df


def calibration_data(y_true: np.ndarray, proba: np.ndarray, class_idx: int, n_bins: int = 10) -> pd.DataFrame:
    """Bitta sinf uchun "one-vs-rest" kalibratsiya nuqtalari (reliability diagram).

    Har bir bin uchun: model shu binda o'rtacha qanday ehtimollik bergan
    (``mean_predicted``) va haqiqatda shu sinf necha foiz uchragan
    (``empirical_freq``). Mukammal kalibratsiyalangan modelda ikkalasi teng
    (diagonal chiziq).
    """
    y_binary = (y_true == class_idx).astype(int)
    empirical_freq, mean_predicted = calibration_curve(
        y_binary, proba[:, class_idx], n_bins=n_bins, strategy="quantile"
    )
    return pd.DataFrame({"mean_predicted": mean_predicted, "empirical_freq": empirical_freq})


def cl_error_analysis(y_true: np.ndarray, proba: np.ndarray, classes: list[str], cl_idx: int = 1) -> dict:
    """`CL` sinfi uchun alohida tahlil: haqiqiy `CL` bo'lgan qatorlarda model
    o'rtacha qanday ehtimollik bergan, va argmax bo'yicha eng ko'p qaysi
    sinfga "adashtirilgan".
    """
    mask = y_true == cl_idx
    n_cl = int(mask.sum())
    mean_proba = proba[mask].mean(axis=0)
    y_pred = proba[mask].argmax(axis=1)
    correct = int((y_pred == cl_idx).sum())
    mistaken_to = pd.Series(y_pred[y_pred != cl_idx]).map(dict(enumerate(classes))).value_counts()

    return {
        "n_cl": n_cl,
        "mean_predicted_proba": dict(zip(classes, mean_proba.round(4))),
        "argmax_correct": correct,
        "argmax_correct_pct": round(100 * correct / n_cl, 1) if n_cl else float("nan"),
        "mistaken_to": mistaken_to.to_dict(),
    }


def slice_log_loss(y_true: np.ndarray, proba: np.ndarray, slice_values, n_classes: int = 3) -> pd.DataFrame:
    """Berilgan ustun (masalan ``Stage`` yoki ``Sex``) qiymatlari bo'yicha guruhlab,
    har bir guruh uchun log loss va guruh hajmini qaytaradi.

    ``slice_values`` — ``y_true``/``proba`` bilan BIR XIL qator tartibidagi
    ketma-ketlik (pd.Series yoki np.ndarray); pozitsion (0-indeksli) moslashtiriladi.
    """
    slice_values = pd.Series(slice_values).reset_index(drop=True)
    rows = []
    for value in sorted(slice_values.dropna().unique()):
        pos = np.flatnonzero((slice_values == value).to_numpy())
        rows.append({
            "guruh": value,
            "n": len(pos),
            "log_loss": overall_log_loss(y_true[pos], proba[pos], n_classes),
        })
    return pd.DataFrame(rows)


def is_improvement_significant(
    loss_a: float, std_a: float, loss_b: float, std_b: float, n_folds: int = 5
) -> bool:
    """Ikki model CV natijasi farqi fold-fold shovqinidan kattaroqmi, degan qo'pol tekshiruv.

    Aniq statistik test emas (fold'lar orasidagi korrelyatsiya, kichik n_folds
    tufayli) — lekin "farq ko'zga tashlanadimi yoki shovqin ichidami" degan
    savolga tezkor javob beradi: farq ikkala modelning fold-standart xatosi
    yig'indisidan katta bo'lsa, "haqiqiy" deb hisoblanadi.
    """
    se_a, se_b = std_a / np.sqrt(n_folds), std_b / np.sqrt(n_folds)
    return abs(loss_a - loss_b) > (se_a + se_b)
