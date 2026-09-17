# cirrhosis-outcomes

Jigar sirrozi bilan og'rigan bemorning kuzatuv oxiridagi holatini (tirik / jigar
transplantatsiyasidan so'ng tirik / vafot etgan) bashorat qiluvchi ma'lumotlar
tahlili loyihasi.

**Musobaqa:** [Multi-Class Prediction of Cirrhosis Outcomes](https://www.kaggle.com/competitions/multiclassificationtask) (Kaggle Playground Series S3E26)
**Asl manba:** Mayo Clinic PBC (birlamchi biliar sirroz) tadqiqoti, UCI Machine Learning Repository

## Masala

`Status` ustunini (3 sinf) bashorat qilish:

| Sinf | Ma'nosi | Ulush |
|---|---|---|
| `C` | tirik (censored) | ~67% |
| `CL` | tirik, jigar transplantatsiyasidan so'ng | ~2.5% |
| `D` | vafot etgan | ~30% |

**Metrika:** multi-class log loss — sinf yorlig'i emas, har bir sinf uchun
ehtimollik topshiriladi (`Status_C`, `Status_CL`, `Status_D`).

## Ma'lumotlar

- `data/raw/train.csv` / `test.csv` — Kaggle musobaqa ma'lumoti (15 000 / 10 000 qator), sun'iy kengaytirilgan.
- `data/raw/cirrhosis.csv` — asl UCI dataset (418 qator), qo'shimcha o'quv ma'lumoti sifatida qo'shiladi.

Fayllar git'ga qo'shilmagan (`.gitignore`) — quyidagi "Ishga tushirish" bo'limida qanday olishni ko'ring.

## Yondashuv qisqacha

Faza-ma-faza jarayon va barcha oraliq qarorlar `TODO.md` da batafsil yozilgan;
chuqur tahlil va tajribalar `notebooks/` papkasida (`01_eda.ipynb` → `07_ensemble.ipynb`).

1. **EDA** — klinik ustunlarning ~43-55%ida bo'sh qiymatlar **tasodifiy emas**,
   blok holida (bir guruh bemorda to'liq tekshiruv bo'lgan, boshqasida yo'q) —
   shuning uchun **to'ldirilmadi**, daraxt modellariga tabiiy holda qoldirildi.
2. **Feature engineering** — kutilmagan topilma: `log1p`, nisbat va klinik ball
   feature'lari LightGBM (daraxt modeli) uchun foyda bermadi — aksincha CV'ni
   yomonlashtirdi (trees monoton transformatsiyalarga o'zgarmas). Yakuniy model
   faqat **sodda kategorial encoding** ("baza") bilan o'qitiladi.
3. **Original UCI data qo'shish** har doim barqaror yordam berdi (~0.003-0.004).
4. **Optuna tuning** (LightGBM, 25 trial): `CV 0.3688 -> 0.3629`.
5. **Ensemble sinaldi, foyda bermadi** — oddiy blend va stacking yakka sozlangan
   LightGBM'dan yomonroq chiqdi (modellar bir-biriga juda o'xshash xato qiladi);
   vaznli blend statistik jihatdan farqlanmadi. **Yakuniy yechim — yakka LightGBM.**

## CV natijalar jadvali

To'liq jadval (barcha oraliq eksperimentlar bilan) — `TODO.md`, 6-bo'lim.

| Model | CV log loss |
|---|---|
| Dummy (pol) | 0.7235 |
| LogisticRegression | 0.4436 |
| HistGradientBoosting | 0.3954 |
| CatBoost | 0.3789 |
| XGBoost | 0.3665 |
| **LightGBM (Optuna tuned) — yakuniy model** | **0.3629** |
| Ensemble (blend/stacking) | 0.3628–0.3853 (yordam bermadi) |

## Repo strukturasi

```
cirrhosis-outcomes/
├── config/config.yaml       # seed, fold soni, yo'llar, model parametrlari
├── data/raw/                # train.csv, test.csv, cirrhosis.csv (git'ga qo'shilmagan)
├── notebooks/                # EDA, tajribalar, baholash, ensemble, o'quv/Kaggle notebook'lari
├── src/
│   ├── config.py             # config.yaml -> dataclass
│   ├── data.py                # yuklash, target encode, original data qo'shish
│   ├── features.py            # feature engineering (guruh bo'yicha yoqib/o'chirish)
│   ├── train.py                # CV, baseline/production model, model saqlash
│   ├── evaluate.py             # sinf bo'yicha log loss, confusion matrix, kalibratsiya
│   ├── ensemble.py             # blend, vazn optimallashtirish, stacking
│   ├── predict.py              # saqlangan modellardan submission
│   └── utils.py                 # seed, logger, timer
├── models/                    # saqlangan fold modellar (.pkl, git'ga qo'shilmagan)
├── outputs/{oof,submissions,figures}/
├── tests/                     # pytest — sintetik ma'lumot bilan, internetsiz ishlaydi
└── Makefile
```

## Ishga tushirish

### 1. Muhitni sozlash

```bash
make setup   # .venv yaratadi (Python 3.12), requirements.txt o'rnatadi
```

### 2. Ma'lumotlarni yuklab olish

```bash
make data
```

Bu buyruq ikki manbadan yuklaydi:
- **Kaggle** (`train.csv`, `test.csv`) — avval [Kaggle API kalitingizni](https://www.kaggle.com/settings/api) sozlang (`~/.kaggle/kaggle.json`, `chmod 600`) va musobaqa qoidalarini saytda qabul qiling.
- **UCI** (`cirrhosis.csv`) — ochiq manba, login shart emas, avtomatik yuklanadi.

### 3. Modelni o'qitish

```bash
make train   # yakuniy LightGBM'ni o'qitadi, models/ ga saqlaydi (~15s)
```

### 4. Bashorat va submission

```bash
make predict   # outputs/submissions/submission.csv yaratadi
```

### Bitta buyruqda hammasi

```bash
make pipeline   # data -> train -> predict
```

### Testlar va lint

```bash
make test   # pytest (sintetik ma'lumot, internetsiz, ~2s)
make lint   # ruff check
```

## Tahlil notebook'lari

| Notebook | Mazmuni |
|---|---|
| `01_eda.ipynb` | Null pattern, sinf balansi, taqsimotlar, drift tahlili |
| `03_experiments.ipynb` | LightGBM/XGBoost/CatBoost, feature ablatsiyasi, Optuna tuning |
| `06_evaluation.ipynb` | Sinf bo'yicha log loss, confusion matrix, kalibratsiya |
| `07_ensemble.ipynb` | Blend/stacking tajribalari, yakuniy qaror |
| `04_kaggle_submission.ipynb` | Kaggle'ga to'g'ridan-to'g'ri yuklanadigan, mustaqil notebook |
| `05_mohirdev_kurs.ipynb` | O'quv maqsadidagi, hikoya uslubidagi tahlil (Mohirdev kursi uchun) |

To'liq faza-ma-faza reja, qarorlar va sabablari — [`TODO.md`](TODO.md).
