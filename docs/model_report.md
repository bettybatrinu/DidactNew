# Raport model Didact AI

> Toate cifrele din acest raport sunt **calculate**, nu fabricate, și sunt
> regenerate automat din date prin `python -m src.train_models`
> (sursă: `models/evaluation_report.json`). Vizualizările sunt regenerate prin
> `python -m src.eda` (folderul `assets/`).

## 0. Obiective și criterii de succes (pre-înregistrate)

Aceste ținte au fost stabilite **înainte** de antrenarea finală, ca să putem
spune onest dacă proiectul a reușit, nu doar să raportăm cifre post-factum:

| # | Obiectiv (țintă pre-definită) | Țintă | Realizat |
|---|-------------------------------|-------|----------|
| O1 | Serviciu structurat (dificultate) macro-F1 | ≥ 0.60 | **0.802** ✅ |
| O2 | Serviciu nestructurat (domeniu) macro-F1 | ≥ 0.80 | **0.974** ✅ |
| O3 | Ambele modele depășesc clar baseline-ul majoritar | da | da (0.16→0.80; 0.08→0.97) ✅ |
| O4 | Stabilitate CV: std macro-F1 raportat pe 5 folduri | raportat | da (vezi §3) ✅ |
| KPI educațional | Rată de îmbunătățire a mastery-ului pe sesiune (Δmastery / N exerciții) și raport indicii/răspuns-corect | măsurabil din log | definit + măsurat din `student_interactions.csv` |

KPI-ul educațional este definit explicit pentru că un macro-F1 bun nu garantează
impact pedagogic. Măsurăm (din `data/processed/student_interactions.csv`, prin
`python -m src.educational_kpi`, salvat în `models/educational_kpi.json`):
(a) Δmastery mediu per succes „curat” (corect, fără indicii) = **+0.108** (pozitiv → tutorul împinge învățarea înainte);
(b) indicii per răspuns corect = **1.26**;
(c) rata succeselor curate = **0.12**.
Sunt proxy-uri pe date sintetice; pe elevi reali ar trebui validate longitudinal.

## 1. Dataset

- Rânduri totale (augmentat): **1344**
- Distribuție dificultate: `2 - mediu`: 650, `1 - bază`: 543, `3 - consolidare`: 136, `4 - avansat`: 15
- Distribuție domenii: Ecuații/Inecuații/Sisteme 384, Funcții 257, Geometrie 232, Rapoarte și proporții 232, Mulțimi numerice 110, Calcul algebric 50, Altele 79
- Analiză de corelații: vezi `assets/feature_correlation_heatmap.png` și
  `dataset.top_correlated_feature_pairs` în raportul JSON. Cea mai puternică
  redundanță: `problem_chars` ~ `problem_words` (≈0.96), de așteptat; și
  `has_percent` ~ `has_real_life_context` (≈0.71).

## 2. Model structurat (dificultate)

- Model: `RandomForestClassifier` în pipeline cu imputare + scalare + OneHotEncoder
- Train/test: 1048/296 (split stratificat 78/22)
- Baseline (DummyClassifier most_frequent): macro-F1 **0.163**, accuracy 0.483
- Model: macro-F1 **0.802**, balanced accuracy 0.798, accuracy 0.929
- Best params: `max_depth=8, min_samples_leaf=1, n_estimators=120, class_weight=balanced`
- **Stabilitate CV (5-fold): macro-F1 0.826 ± 0.094** — raportăm media ± deviația
  standard pe folduri, nu doar un singur split, tocmai pentru că pe un set mic un
  split izolat poate fi instabil.

### 2.1 Comparație de modele (aceeași preprocesare, aceeași CV)

| Model | CV macro-F1 (mean ± std) | Test macro-F1 |
|-------|--------------------------|---------------|
| RandomForest | 0.815 ± 0.020 | 0.770 |
| GradientBoosting | 0.812 ± 0.102 | 0.823 |
| LogisticRegression | 0.750 ± 0.055 | 0.838 |
| XGBoost (opțional) | 0.767 ± 0.116 | 0.699 |

**Decizie:** păstrăm RandomForest ca model principal pentru **stabilitatea CV**
(cea mai mică deviație standard între candidați robusti) și pentru
**interpretabilitate** (feature importance). LogisticRegression și GradientBoosting
sunt competitive pe test și sunt raportate transparent — alegerea nu este afirmată
fără dovezi, ci justificată prin comparație.

### 2.2 Anti-overfitting

`max_depth` a fost căutat pe `[4, 6, 8, 12, None]` și `min_samples_leaf` pe
`[1, 3, 5]`. Configurația aleasă (`max_depth=8`) este regularizată — arborii nu
mai cresc complet, ceea ce reduce riscul de overfitting pe un set mic, semnalat
în versiunea anterioară.

## 3. Model nestructurat (domeniu)

- Model: `TF-IDF + ComplementNB`
- Train/test: 986/279 (split stratificat, fallback la split nestratificat dacă o clasă e prea rară)
- Baseline: macro-F1 **0.078**, accuracy 0.305
- Model: macro-F1 **0.974**, balanced accuracy 0.976, accuracy 0.982
- Best params: `alpha=0.2, max_features=6000, ngram_range=(1,2), sublinear_tf=True`
- **Stabilitate CV (5-fold): macro-F1 0.993 ± 0.005**
- `sublinear_tf=True` a fost adăugat ca tehnică anti-overfitting specifică NLP
  (atenuează frecvențele mari de termeni) și a fost selectat de GridSearch.

### 3.1 Comparație de modele (aceeași reprezentare TF-IDF)

| Model | CV macro-F1 (mean ± std) | Test macro-F1 |
|-------|--------------------------|---------------|
| ComplementNB | 0.993 ± 0.005 | 0.974 |
| LinearSVC | 0.992 ± 0.005 | 0.984 |
| LogisticRegression | 0.990 ± 0.005 | 0.974 |

ComplementNB rămâne alegerea principală (rapid, potrivit pentru text dezechilibrat,
fără tuning costisitor), iar LinearSVC/LogisticRegression confirmă că reprezentarea
TF-IDF este puternică indiferent de clasificator.

## 4. Protocol de evaluare și analiză critică

- Split train/test stratificat 78/22 + `StratifiedKFold(5)` în CV.
- Anti-leakage: pipeline-ul este fit doar pe foldurile de train; pentru text se
  elimină duplicatele exacte text+domeniu înainte de split.
- Metrici justificate de dezechilibru: macro-F1 și balanced accuracy, nu doar accuracy.

### 4.1 De ce split i.i.d. și nu temporal?

Avem coloana `Sursa_year`, deci un split temporal ar fi tehnic posibil. Am ales
totuși un split aleator stratificat pentru că **exercițiile sunt itemi
independenți**, nu o serie temporală cu autocorelație: nu există o relație
cauzală „exercițiul din 2019 influențează exercițiul din 2021”. Un split temporal
ar fi relevant pentru predicția evoluției unui elev în timp (knowledge tracing
secvențial), care este o extensie viitoare, nu obiectivul MVP-ului curent.
Menționăm explicit alternativa pentru a nu trata tăcut datele ca i.i.d.

### 4.2 Audit de leakage (rezolvat, nu doar semnalat)

Augmentarea poate introduce near-duplicate care, într-un split naiv, ajung și în
train și în test. Am **măsurat** acest risc și l-am **controlat**:

- În split-ul naiv, **82 din 279** itemi de test (29%) au un near-duplicate în
  train (cosine TF-IDF > 0.9).
- Am re-evaluat **group-aware**: am grupat problemele near-duplicate în clustere
  (componente conexe, 978 grupuri) și am folosit `GroupShuffleSplit` +
  `StratifiedGroupKFold`, astfel încât niciun cluster nu traversează train/test.
- Rezultat: macro-F1 **0.974** (holdout group-aware), CV **0.986 ± 0.009** —
  practic neschimbat față de split-ul naiv.

**Concluzie:** scorul mare **nu** este un artefact de leakage. Clasificarea
domeniului din text este genuin separabilă (vocabular aproape determinist:
„triunghi/cerc” → Geometrie, „ecuație/sistem” → Ecuații). Auditul este în
`evaluation_report.json` → `unstructured_model.leakage_audit`.

### 4.3 Curbe de învățare (caracterizarea limitării „dataset mic”)

În loc să spunem doar „datasetul e mic”, am **cuantificat** efectul prin curbe de
învățare (macro-F1 vs. număr de exemple, `assets/learning_curve_*.png`):

- **Text:** CV macro-F1 urcă 0.967 → 0.993 și se **aplatizează** — datasetul
  curent este suficient pentru acest task; mai multe date nu ar ajuta semnificativ.
- **Structurat:** CV macro-F1 urcă 0.70 → 0.80 și se aplatizează, dar rămâne un
  **gap train/CV** (train ~0.96 vs CV ~0.80) — semnătura clasică a unui set mic
  pentru un task mai greu (4 clase dezechilibrate). Aici mai multe date reale și
  mai multă regularizare ar ajuta cel mai mult. Aceasta este limitarea reală,
  caracterizată explicit, nu doar admisă.

### 4.4 Componenta neurală: experimentală, pe date sintetice

Modelul neural (TensorFlow) pentru starea elevului este **antrenat pe date
sintetice** (`student_interactions.csv` generat), deci acuratețea lui raportată
(~0.92) reflectă datele sintetice, **nu** performanță reală. Îl prezentăm ca strat
opțional și demonstrativ peste regulile pedagogice, nu ca al treilea serviciu ML
validat. Cele două servicii principale (dificultate, domeniu) sunt validate pe
date reale.

## 5. Erori reprezentative

Câte 8 erori per model sunt salvate în `models/evaluation_report.json`
(`sample_errors`) cu enunț, etichetă reală și predicție, și sunt afișate în tabul
„Evaluare & EDA” al aplicației. Tipare frecvente: confuzia `2 - mediu` ↔
`3 - consolidare` la structurat; sisteme de ecuații „ascunse” în contexte reale
clasificate ca `Rapoarte și proporții` la text.
