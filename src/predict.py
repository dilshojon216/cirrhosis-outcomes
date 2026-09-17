"""Saqlangan fold modellardan ``test.csv`` uchun ``submission.csv`` generatsiya qilish.

Bu modul HECH QANDAY modelni o'qitmaydi — faqat ``src/train.py`` ning
``train_final_model()`` (``python -m src.train`` / ``make train``) allaqachon
``models/`` ga saqlagan fold modellarini o'qiydi. Model topilmasa, aniq xato
bilan avval trening ishga tushirish kerakligini aytadi.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.train import (
    FINAL_FEATURE_GROUPS,
    FINAL_MODEL_NAME,
    get_numeric_feature_cols,
    load_fold_models,
    prepare_train_test,
)
from src.utils import get_logger, timer

logger = get_logger(__name__)


def predict_proba_bagged(models: list, X: pd.DataFrame, eps: float = 1e-7) -> np.ndarray:
    """Har bir fold modelining bashoratini o'rtachalaydi ("5-fold bagging"),
    so'ng raqamli chegaralaydi (``numpy.clip``) — bitta ehtimollik aynan 0/1
    bo'lib qolib, ``log(0) = -inf`` bilan log loss'ni "portlatib" yubormasligi
    uchun (``04_kaggle_submission.ipynb``/``05_mohirdev_kurs.ipynb`` bilan bir xil naqsh).
    """
    proba = np.mean([m.predict_proba(X) for m in models], axis=0)
    proba = np.clip(proba, eps, 1 - eps)
    return proba / proba.sum(axis=1, keepdims=True)


@timer
def generate_submission(
    cfg: Config, model_name: str = FINAL_MODEL_NAME, out_name: str = "submission.csv"
) -> Path:
    """``test.csv`` uchun bashorat qilib, ``outputs/submissions/<out_name>`` ga saqlaydi."""
    _, test = prepare_train_test(cfg, groups=FINAL_FEATURE_GROUPS)
    feature_cols = get_numeric_feature_cols(test, cfg)

    models = load_fold_models(cfg, model_name)
    proba = predict_proba_bagged(models, test[feature_cols])

    submission = pd.DataFrame(proba, columns=[f"Status_{c}" for c in cfg.data.classes], index=test.index)
    submission.index.name = cfg.data.id_col
    submission = submission.reset_index()

    cfg.paths.submissions_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.paths.submissions_dir / out_name
    submission.to_csv(out_path, index=False)
    logger.info("generate_submission: %s (%s)", out_path, submission.shape)
    return out_path


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    path = generate_submission(cfg)
    print(f"Submission saqlandi: {path}")
