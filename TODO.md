# Sirroz bemorlari omon qolish prognozi — loyiha rejasi

> Kaggle musobaqa: https://www.kaggle.com/competitions/multiclassificationtask/overview
> Asos: Kaggle Playground Series S3E26 — *Multi-Class Prediction of Cirrhosis Outcomes*
> Repo nomi: `cirrhosis-outcomes`

---

## 1. Masala

Jigar sirrozi bilan og'rigan bemorning kuzatuv oxiridagi holatini bashorat qilish.
Target — `Status`, 3 ta sinf:

| Sinf | Ma'nosi | Taxminiy ulush |
|---|---|---|
| `C` | bemor tirik (censored) | ~63% |
| `CL` | tirik, jigar transplantatsiyasidan so'ng | ~3% |
| `D` | vafot etgan | ~34% |

**Metrika:** multi-class log loss → sinf yorlig'i emas, **ehtimolliklar** topshiriladi.

**Submission formati:**

```csv
id,Status_C,Status_CL,Status_D
7905,0.71,0.02,0.27
```

> ⚠️ Metrikani Kaggle sahifasidagi *Evaluation* bo'limidan tasdiqlang — Mohirdev versiyasida o'zgargan bo'lishi mumkin.

---

## 2. Ma'lumotlar

Asli UCI / Mayo Clinic PBC tadqiqoti, Kaggle uni sintetik generatsiya bilan kengaytirgan.

**Raqamli ustunlar**

| Ustun | Izoh |
|---|---|
| `N_Days` | kuzatuv davomiyligi (kun) — eng kuchli signal |
| `Age` | yosh, kunlarda |
| `Bilirubin` | o'ngga kuchli qiya |
| `Cholesterol` | null'lar ko'p bo'lishi mumkin |
| `Albumin` | |
| `Copper` | o'ngga qiya |
| `Alk_Phos` | o'ngga juda kuchli qiya |
| `SGOT` | o'ngga qiya |
| `Tryglicerides` | |
| `Platelets` | |
| `Prothrombin` | |

**Kategorial ustunlar**

| Ustun | Qiymatlar |
|---|---|
| `Drug` | D-penicillamine / Placebo |
| `Sex` | M / F |
| `Ascites` | N / Y |
| `Hepatomegaly` | N / Y |
| `Spiders` | N / Y |
| `Edema` | N / S / Y |
| `Stage` | 1–4 (ordinal) |

---

## 3. Asosiy qiyinchiliklar

1. **`CL` sinfi juda kam (~3%).** Log loss'da bu sinfga haddan ortiq ishonchli bashorat berish jarimani keskin oshiradi. Kalibratsiya muhim.
2. **Skewed klinik ustunlar.** `Bilirubin`, `Copper`, `Alk_Phos`, `SGOT` — `log1p` odatda yordam beradi.
3. **`N_Days` dominantligi.** Model deyarli faqat shunga tayanib qolishi mumkin — feature importance'ni tekshiring.
4. **Sintetik data.** Asl UCI datasetini qo'shimcha train sifatida qo'shish odatda CV'ni yaxshilaydi, lekin validatsiya faqat musobaqa train'ida bo'lishi kerak.
5. **CV ↔ LB farqi.** Stratifikatsiya `Status` bo'yicha, seed qat'iy.

---

## 4. Repo strukturasi

```
cirrhosis-outcomes/
├── README.md
├── requirements.txt
├── .gitignore
├── Makefile
│
├── config/
│   └── config.yaml          # seed, fold soni, model paramlari, yo'llar
│
├── data/
│   ├── raw/                 # train.csv, test.csv, sample_submission.csv
│   │   └── cirrhosis.csv    # asl UCI dataset
│   ├── processed/           # tozalangan parquet
│   └── .gitignore
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py            # config.yaml -> dataclass
│   ├── data.py              # yuklash, original data merge, target encode
│   ├── features.py          # log1p, encoding, yangi feature'lar
│   ├── train.py             # StratifiedKFold CV, OOF, model saqlash
│   ├── evaluate.py          # sinf bo'yicha log loss, confusion, kalibratsiya
│   ├── predict.py           # test -> submission.csv
│   └── utils.py             # seed_everything, logger, timer
│
├── models/                  # fold_0.pkl ... (gitignore)
├── outputs/
│   ├── oof/
│   ├── submissions/
│   └── figures/
│
├── tests/
│   ├── test_features.py
│   └── test_pipeline.py
│
└── .github/workflows/ci.yml
```

---

## 5. TODO

### Faza 0 — Skelet

- [x] `cirrhosis-outcomes` repo yaratish, `git init`
- [x] `.gitignore` (data/, models/, outputs/, `.ipynb_checkpoints`, `__pycache__`, `.venv`)
- [x] `requirements.txt` — pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, optuna, matplotlib, seaborn, pyyaml, pytest
- [x] Virtual muhit + `pip install -r requirements.txt`
- [x] `config/config.yaml` — `seed: 42`, `n_folds: 5`, yo'llar, model paramlari
- [x] `src/utils.py` — `seed_everything()`, logger, timer dekoratori
- [ ] Kaggle'dan `train.csv`, `test.csv`, `sample_submission.csv` yuklab olish
- [x] Asl UCI `cirrhosis.csv` ni yuklab olish

### Faza 1 — EDA

- [x] Shape, dtypes, `df.info()`, null count har bir ustun bo'yicha
- [x] `Status` taqsimoti — sinf disbalansini raqamda ko'rish
- [x] Raqamli ustunlar histogrammasi — qaysilari skewed
- [x] Kategorial ustunlar bo'yicha `Status` nisbati (crosstab)
- [x] Korrelyatsiya matritsasi (raqamli ustunlar)
- [x] `N_Days` ning har bir sinf bo'yicha taqsimoti (boxplot)
- [x] Outlier'larni belgilash (`Bilirubin`, `Alk_Phos`, `Copper`)
- [x] Train vs test taqsimotini solishtirish (drift bormi)
- [x] Grafiklarni `outputs/figures/` ga saqlash
- [x] EDA xulosalarini notebook oxirida 5–7 punktda yozib qo'yish

### Faza 2 — Data qatlami

- [x] `src/data.py`: `load_raw()` — train/test o'qish, `id` ni index qilish
- [x] `encode_target()` — `C/CL/D` → `0/1/2`, teskari mapping ham
- [x] `load_original()` — UCI datasetni musobaqa ustunlariga moslashtirish
- [x] `merge_original()` — flag ustuni bilan (`is_original`), faqat train'ga
- [x] Null strategiyasi: qoldirish (LightGBM o'zi hal qiladi) yoki median — qaror yozib qo'yish — **qoldirish** tanlandi (sabab: `src/data.py` docstring, EDA #1 — null'lar blok holida MNAR, median soxta signal beradi)
- [x] `processed/` ga parquet saqlash

### Faza 3 — Feature engineering

- [x] `src/features.py`: `add_log_features()` — `Bilirubin`, `Copper`, `Alk_Phos`, `SGOT`, `Cholesterol`
- [x] `Age` ni yilga o'tkazish (`Age / 365.25`)
- [x] Kategorial encoding — ordinal (`Stage`, `Edema`) va binary (`Y/N` → 1/0) — `Sex`/`Drug` ham binary sifatida qo'shildi; xom ustunlar CatBoost uchun saqlanadi
- [x] Nisbat feature'lar: `Bilirubin/Albumin`, `SGOT/Alk_Phos`, `Copper/Age_years`
- [x] Klinik ball: `Ascites + Hepatomegaly + Spiders + Edema` yig'indisi
- [x] **fit faqat train'da, transform ikkalasida** — leakage'ni oldini olish — `FeatureEngineer` klassi (hozircha stateless, kelajakdagi encoder'lar uchun disiplina)
- [x] Har bir feature guruhini alohida yoqib/o'chirib CV'da tekshirish — `build_features(df, cfg, groups=(...))`, qo'lda tekshirildi
- [x] **Qo'shimcha (EDA #1 asosida):** `is_full_labs` / `n_missing_clinical` — blok-missing indikatori

### Faza 4 — Baseline

- [x] `src/train.py`: `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` — faqat `is_original==0` qatorlarda, original qatorlar hech qachon validatsiyaga tushmaydi
- [x] Dummy baseline — sinf chastotalari bilan log loss (pol raqami)
- [x] `LogisticRegression` baseline
- [x] `HistGradientBoostingClassifier` baseline
- [x] OOF ehtimolliklarni `outputs/oof/` ga saqlash
- [x] Har bir baseline CV log loss'ini jadvalga yozish (6-bo'lim)
- [ ] Birinchi submission yuborib, CV ↔ LB farqini o'lchash — `outputs/submissions/baseline_hgb.csv` tayyorlandi (HGB, to'liq train'ga fit qilingan), lekin **Kaggle'ga yuklash foydalanuvchi tomonidan qilinishi kerak** (Kaggle API kaliti bu muhitda sozlanmagan):
  ```bash
  .venv/bin/kaggle competitions submit -c multiclassificationtask -f outputs/submissions/baseline_hgb.csv -m "Faza 4 baseline: HGB"
  ```

### Faza 5 — Model tuning

- [x] LightGBM — `objective='multiclass'`, `num_class=3`, `metric='multi_logloss'`
- [x] XGBoost — `objective='multi:softprob'`
- [x] CatBoost — kategoriallarni to'g'ridan-to'g'ri berish (`cat_features`, `NaN`→`"missing"`)
- [x] Har biriga `early_stopping` fold ichida (`run_cv(fit_kwargs_fn=...)`)
- [x] Optuna bilan tuning (kamida LightGBM uchun) — 25 trial, `CV 0.36880 -> 0.36288`, `config.yaml` yangilandi
- [x] `class_weight` / `sample_weight` ni `CL` uchun sinab ko'rish — yordam bermadi (umumiy CV yomonlashdi), lekin natija jadvalda saqlandi
- [x] Original datani qo'shgan va qo'shmagan holatni solishtirish — barqaror yordam berdi (~0.003-0.004), `use_original: true` ga o'zgartirildi
- [x] Feature importance grafigi — `N_Days` dominantligini tekshirish — **dominant emas**, `Platelets`/`Age` bilan bir xil darajada
- [x] **Kutilmagan topilma:** `src/features.py` dagi qo'shimcha feature'lar (log1p/nisbat/klinik ball) daraxt modellari uchun CV'ni yomonlashtirdi — "baza" (faqat categorical encoding) eng yaxshi natija berdi
- [x] To'liq tahlil: `notebooks/03_experiments.ipynb` (bajarilgan, xatosiz)
- [x] **Qo'shimcha (rejadan tashqari, foydalanuvchi so'rovi bo'yicha):** `notebooks/04_kaggle_submission.ipynb` — Kaggle'ga to'g'ridan-to'g'ri yuklanadigan, mustaqil (src/ import qilmaydi), sodda va tushunarli (Mohirdev kursi uchun) yagona notebook; lokal ma'lumotda tekshirildi, xuddi shu CV=0.36288 natijani qaytardi

### Faza 6 — Evaluation

- [x] `src/evaluate.py`: umumiy va **sinf bo'yicha** log loss — `overall_log_loss()`, `per_class_log_loss()`
- [x] Confusion matrix (argmax bo'yicha) — `confusion_matrix_df()`
- [x] Kalibratsiya egri chizig'i har bir sinf uchun — `calibration_data()`; barcha 3 sinf yaxshi kalibratsiyalangan (diagonalga yaqin)
- [x] `CL` sinfida model qanday xato qilayotganini alohida tahlil — `cl_error_analysis()`: argmax bo'yicha atigi 21.8% to'g'ri, lekin ehtimollik taqsimoti yaxshi kalibratsiyalangan (model xatosi emas, muammoning tabiati)
- [x] Slice tahlili: `Stage` va `Sex` bo'yicha log loss — `slice_log_loss()`; Stage bilan monoton oshadi (0.18→0.45), Sex'da erkaklar guruhida yuqoriroq (kichik n=608, ehtiyotkorlik bilan)
- [x] CV standart og'ishini yozish — yaxshilanish shovqindan kattami — `is_improvement_significant()`: XGBoost vs tuned LightGBM farqi (0.36646→0.36288) shovqin ichida (statistik jihatdan ajratib bo'lmaydi)
- [x] To'liq tahlil: `notebooks/06_evaluation.ipynb` (bajarilgan, xatosiz, 5 ta grafik `outputs/figures/` ga saqlandi)

### Faza 7 — Ensemble

- [x] 3 model OOF'ini yuklab, oddiy o'rtacha blend — `0.36574`, yakka `lgbm_tuned` (`0.36288`)dan **yomonroq** (`catboost` zaifligi tortadi)
- [x] Vaznli blend — vaznlarni OOF'da optimallashtirish (`scipy.optimize`) — `lgbm=0.90/xgb=0.10/catboost=0.00`, natija `0.36284` (statistik ahamiyatsiz farq)
- [x] Stacking: OOF ehtimolliklar ustiga LogisticRegression meta-model — `0.38525`, **eng yomon natija** (feature'lar juda korrelyatsiyalangan)
- [x] Eng yaxshi variantni tanlash va sababi bilan yozib qo'yish — **yakka `lgbm_tuned` tanlandi**: blend statistik farqlanmaydi, stacking/oddiy blend yomonlashtiradi, soddalik ustuvor
- [x] Yakuniy submission generatsiyasi — `outputs/submissions/final_lgbm.csv` (5-fold bagging, `np.clip` bilan)
- [x] To'liq tahlil: `notebooks/07_ensemble.ipynb` (bajarilgan, xatosiz)
- [x] **Bug fix (ishlab chiqish paytida topildi):** `prepare_train_test()` doim barcha feature guruhini qo'llardi — g'olib "baza" konfiguratsiya (`groups=("categorical",)`) uchun `groups` parametri qo'shildi, aks holda yakuniy submission noto'g'ri feature to'plamida o'qitilardi (qayta tekshiruvda `0.36745` chiqib, `0.36288` bilan mos kelmagani orqali aniqlandi)

### Faza 8 — Production tozalash

- [x] `src/predict.py` — saqlangan fold modellardan submission (`load_fold_models`, `predict_proba_bagged`, `generate_submission`)
- [x] `src/train.py`: `save_fold_models()`/`load_fold_models()`/`train_final_model()` qo'shildi — production model `models/lgbm_final_fold*.pkl` ga saqlanadi
- [x] `Makefile`: `make setup`, `make data`, `make train`, `make predict`, `make test`, `make lint`, `make pipeline`
- [x] Bitta buyruqdan to'liq pipeline ishlashini tekshirish — `make train` + `make predict` qo'lda ishga tushirilib tasdiqlandi (`submission.csv` == `07_ensemble.ipynb` natijasi bilan **bit-bit bir xil**)
- [x] `tests/test_features.py` — transformatsiya logikasi (17 test, `conftest.py` fixture'lari bilan)
- [x] `tests/test_pipeline.py` — kichik sintetik sample'da to'liq pipeline smoke test (4 test)
- [x] `.github/workflows/ci.yml` — ruff + pytest (Python 3.12, haqiqiy ma'lumotsiz — sintetik fixture'lar bilan)
- [x] `README.md`: masala, data, yondashuv, CV jadvali, natija, qanday ishga tushirish
- [x] Repo'ni begona odam nol kontekst bilan ishga tushira olishini tekshirish — `python3.12 -m venv` + `pip install -r requirements.txt` (README'dagi aynan `make setup` yo'li, `uv` EMAS) alohida, toza `/tmp` venv'da sinaldi: 21/21 test o'tdi, `ruff check` xatosiz

---

## 6. Natijalar jadvali

Har bir eksperimentdan keyin shu jadvalni to'ldirib boring:

| # | Model | Feature'lar | Original data | CV log loss | LB log loss | Izoh |
|---|---|---|---|---|---|---|
| 0 | Dummy | — | yo'q | 0.72352 ± 0.00043 | | pol raqami (nazariy: sinf chastotalari entropiyasi ≈0.722, mos keladi) |
| 1 | LogReg | baza (30 raqamli ustun, median impute) | yo'q | 0.44363 ± 0.01052 | | |
| 2 | HGB | baza (30 raqamli ustun, NaN'siz impute) | yo'q | 0.39543 ± 0.00844 | | `outputs/submissions/baseline_hgb.csv` tayyor, Kaggle'ga hali yuklanmadi |
| 3 | LightGBM | baza (18 feat, categorical encoding) | yo'q | 0.37235 ± 0.00891 | | |
| 4 | LightGBM | + log (23 feat) | yo'q | 0.37372 ± 0.00939 | | log1p daraxt modeliga yordam bermadi (yomonlashdi) |
| 5 | LightGBM | + log (23 feat) | ha | 0.37003 ± 0.00900 | | |
| 6 | LightGBM | full (30 feat, barcha guruh) | yo'q | 0.37636 ± 0.00918 | | eng ko'p feature, eng yomon natija (daraxt modeli uchun) |
| 7 | LightGBM | full (30 feat) | ha | 0.37293 ± 0.00962 | | |
| 8 | LightGBM | baza (18 feat) | ha | 0.36880 ± 0.00853 | | eng yaxshi feature to'plami (tuning'gacha) |
| 9 | LightGBM | baza, `class_weight="balanced"` | ha | 0.40050 | | `CL` uchun yaxshi, umumiy uchun yomon (rad etildi, izoh: 03_experiments.ipynb §2) |
| 10 | LightGBM | baza, `class_weight={0:1,1:10,2:1}` | ha | 0.38379 | | xuddi shunday — umumiy metrikaga mos emas |
| 11 | XGBoost | baza (18 feat), early stopping | ha | 0.36646 ± 0.00912 | | ikkinchi eng yaxshi yagona model |
| 12 | CatBoost | baza (18 feat), xom kategorial | ha | 0.37894 ± 0.00930 | | |
| 13 | **LightGBM (Optuna, 25 trial)** | **baza (18 feat)** | **ha** | **0.36288 ± 0.00926** | | **eng yaxshi yagona model** — `config.yaml` ga yozildi |
| 14 | Blend | — | ha | | | Faza 7 |

---

## 7. Qoidalar

1. Bitta eksperimentda bitta narsani o'zgartiring.
2. Seed hamma joyda qat'iy — solishtirishdan oldin.
3. CV `train.csv` ustida, original data faqat train fold'ga qo'shiladi.
4. Har bir submission uchun qaysi kod commit'idan chiqqanini yozib qo'ying.
5. Yomon natijali eksperimentlarni ham jadvalda qoldiring — keyingi qarorlarni ular tushuntiradi.
6. `CL` sinfida ehtimollik hech qachon 0 ga yaqin bo'lmasin — log loss uni jazolaydi.
