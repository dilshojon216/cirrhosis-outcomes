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

- [ ] `src/train.py`: `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`
- [ ] Dummy baseline — sinf chastotalari bilan log loss (pol raqami)
- [ ] `LogisticRegression` baseline
- [ ] `HistGradientBoostingClassifier` baseline
- [ ] OOF ehtimolliklarni `outputs/oof/` ga saqlash
- [ ] Har bir baseline CV log loss'ini jadvalga yozish
- [ ] Birinchi submission yuborib, CV ↔ LB farqini o'lchash

### Faza 5 — Model tuning

- [ ] LightGBM — `objective='multiclass'`, `num_class=3`, `metric='multi_logloss'`
- [ ] XGBoost — `objective='multi:softprob'`
- [ ] CatBoost — kategoriallarni to'g'ridan-to'g'ri berish
- [ ] Har biriga `early_stopping` fold ichida
- [ ] Optuna bilan tuning (kamida LightGBM uchun)
- [ ] `class_weight` / `sample_weight` ni `CL` uchun sinab ko'rish
- [ ] Original datani qo'shgan va qo'shmagan holatni solishtirish
- [ ] Feature importance grafigi — `N_Days` dominantligini tekshirish

### Faza 6 — Evaluation

- [ ] `src/evaluate.py`: umumiy va **sinf bo'yicha** log loss
- [ ] Confusion matrix (argmax bo'yicha)
- [ ] Kalibratsiya egri chizig'i har bir sinf uchun
- [ ] `CL` sinfida model qanday xato qilayotganini alohida tahlil
- [ ] Slice tahlili: `Stage` va `Sex` bo'yicha log loss
- [ ] CV standart og'ishini yozish — yaxshilanish shovqindan kattami

### Faza 7 — Ensemble

- [ ] 3 model OOF'ini yuklab, oddiy o'rtacha blend
- [ ] Vaznli blend — vaznlarni OOF'da optimallashtirish (`scipy.optimize`)
- [ ] Stacking: OOF ehtimolliklar ustiga LogisticRegression meta-model
- [ ] Eng yaxshi variantni tanlash va sababi bilan yozib qo'yish
- [ ] Yakuniy submission generatsiyasi

### Faza 8 — Production tozalash

- [ ] `src/predict.py` — saqlangan fold modellardan submission
- [ ] `Makefile`: `make setup`, `make train`, `make predict`
- [ ] Bitta buyruqdan to'liq pipeline ishlashini tekshirish
- [ ] `tests/test_features.py` — transformatsiya logikasi
- [ ] `tests/test_pipeline.py` — kichik sample'da smoke test
- [ ] `.github/workflows/ci.yml` — ruff + pytest
- [ ] `README.md`: masala, data, yondashuv, CV jadvali, natija, qanday ishga tushirish
- [ ] Repo'ni begona odam nol kontekst bilan ishga tushira olishini tekshirish

---

## 6. Natijalar jadvali

Har bir eksperimentdan keyin shu jadvalni to'ldirib boring:

| # | Model | Feature'lar | Original data | CV log loss | LB log loss | Izoh |
|---|---|---|---|---|---|---|
| 0 | Dummy | — | yo'q | | | pol raqami |
| 1 | LogReg | baza | yo'q | | | |
| 2 | LightGBM | baza | yo'q | | | |
| 3 | LightGBM | + log | yo'q | | | |
| 4 | LightGBM | + log | ha | | | |
| 5 | XGBoost | eng yaxshi | ha | | | |
| 6 | CatBoost | eng yaxshi | ha | | | |
| 7 | Blend | — | ha | | | |

---

## 7. Qoidalar

1. Bitta eksperimentda bitta narsani o'zgartiring.
2. Seed hamma joyda qat'iy — solishtirishdan oldin.
3. CV `train.csv` ustida, original data faqat train fold'ga qo'shiladi.
4. Har bir submission uchun qaysi kod commit'idan chiqqanini yozib qo'ying.
5. Yomon natijali eksperimentlarni ham jadvalda qoldiring — keyingi qarorlarni ular tushuntiradi.
6. `CL` sinfida ehtimollik hech qachon 0 ga yaqin bo'lmasin — log loss uni jazolaydi.
