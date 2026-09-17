"""OOF ehtimolliklarni birlashtirish: oddiy/vaznli blend va stacking (Faza 7).

Bu modul faqat allaqachon o'qitilgan modellarning OOF (out-of-fold)
ehtimolliklari ustida ishlaydi — yangi model o'qitmaydi. Yakuniy test
bashoratini generatsiya qilish uchun asosiy modellarni qayta o'qitish
kerak (``notebooks/07_ensemble.ipynb`` da qilinadi, ``src/train.py``
funksiyalari orqali).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from src.utils import get_logger

logger = get_logger(__name__)


def blend_proba(probas: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Vaznli o'rtacha: ``sum(w_i * probas[i])``. Vaznlar 1ga yig'ilishi kerak
    (shunda natija ham har bir qatorda 1ga yig'iladigan haqiqiy ehtimollik bo'ladi)."""
    weights = np.asarray(weights, dtype=float)
    stacked = np.stack(probas, axis=0)  # (k_model, n_qator, n_sinf)
    return np.tensordot(weights, stacked, axes=(0, 0))


def optimize_blend_weights(
    y_true: np.ndarray, probas: list[np.ndarray], n_classes: int = 3
) -> np.ndarray:
    """``scipy.optimize`` (SLSQP) bilan CV log loss'ni minimallashtiradigan vaznlarni topadi.

    Cheklov: vaznlar manfiy emas va 1ga yig'iladi (convex combination) — shu
    cheklov ostida log loss vaznlarga nisbatan convex (proba_blend vaznlarga
    nisbatan chiziqli, log_loss esa proba'ga nisbatan convex), shuning uchun
    bitta boshlanish nuqtasi (teng vaznlar) yetarli — global minimumga yaqinlashadi.
    """
    k = len(probas)

    def objective(w: np.ndarray) -> float:
        proba = blend_proba(probas, w)
        return log_loss(y_true, proba, labels=list(range(n_classes)))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * k
    x0 = np.full(k, 1.0 / k)

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        logger.warning(
            "optimize_blend_weights: optimallashtirish muvaffaqiyatsiz (%s) — teng vaznga qaytiladi",
            result.message,
        )
        return x0
    return result.x


def stacking_oof(
    y_true: np.ndarray,
    probas: list[np.ndarray],
    n_folds: int = 5,
    seed: int = 42,
    n_classes: int = 3,
) -> tuple[np.ndarray, list[float]]:
    """LogisticRegression meta-model — asosiy modellarning OOF ehtimolliklarini
    (``k_model * n_classes`` ustun) feature sifatida ishlatadi.

    Muhim: bu OOF ehtimolliklar asosiy modellar uchun allaqachon out-of-fold,
    lekin meta-modelning o'zi ham shu ma'lumotga fit qilib, o'sha ma'lumotda
    baholansa — bu leakage bo'lardi. Shuning uchun meta-model ustida ALOHIDA
    (nested) StratifiedKFold CV yuritiladi — meta-model ham faqat o'zi
    ko'rmagan qatorlarda baholanadi.
    """
    X_meta = np.concatenate(probas, axis=1)  # (n, k_model * n_classes)
    oof_stack = np.full((len(y_true), n_classes), np.nan)
    fold_losses = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_idx, val_idx in skf.split(X_meta, y_true):
        meta = LogisticRegression(max_iter=2000, random_state=seed)
        meta.fit(X_meta[fit_idx], y_true[fit_idx])
        proba = meta.predict_proba(X_meta[val_idx])
        oof_stack[val_idx] = proba
        fold_losses.append(log_loss(y_true[val_idx], proba, labels=list(range(n_classes))))

    return oof_stack, fold_losses
