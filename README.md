# Didact AI - tutor adaptiv de matematică

Didact AI este un MVP pentru Olimpiada Națională de Inteligență Artificială: o aplicație Streamlit care ajută elevul să exerseze matematică fără să primească automat soluția completă. Sistemul combină reguli pedagogice controlabile cu două servicii ML reale și demonstrabile.

## Obiective și criterii de succes (pre-înregistrate)

Aceste ținte au fost stabilite **înainte** de antrenarea finală, ca să putem
judeca onest succesul, nu doar să raportăm cifre după fapt:

- **O1 — dificultate (structurat):** macro-F1 ≥ 0.60 → realizat **0.802**.
- **O2 — domeniu (text):** macro-F1 ≥ 0.80 → realizat **0.974**.
- **O3 — depășire baseline:** ambele modele bat clar baseline-ul majoritar (0.16→0.80; 0.08→0.97).
- **O4 — stabilitate:** raportăm macro-F1 CV ca medie ± deviație standard pe 5 folduri.
- **KPI educațional:** rata de îmbunătățire a mastery-ului (Δmastery per exercițiu corect fără indicii) și raportul indicii/răspuns-corect, măsurate din `data/processed/student_interactions.csv`. Acest KPI există pentru că un macro-F1 bun nu garantează impact pedagogic.

Quick Docker / Cloud-safe defaults
----------------------------------
This repository includes a Dockerfile that builds on `python:3.11-slim` and installs the default `requirements.txt` (cloud-safe, no TensorFlow). To build and run the app locally in Docker:

```bash
./run_docker.sh
# then open http://localhost:8501
```

Notes:
- The Docker image uses Python 3.11 by default.
- For the TensorFlow-enabled environment, install `requirements-full.txt` instead.
- If the package installation fails, check the build output; common causes are unavailable TensorFlow wheels or missing system libraries.


## Decizia despre dataset

Datasetul furnizat este **utilizabil pentru o demonstrație competitivă**, dar nu este încă suficient pentru producție. Setul brut are 489 exerciții (477 cu temă + dificultate, 51 duplicate exacte, 12 rânduri fără etichete), iar pentru antrenare îl augmentăm la **1344 rânduri**. Nu recomand înlocuirea lui acum, pentru că este în limba română, este aliniat la matematica de examen și conține exact câmpurile necesare pentru cele două servicii ML: textul problemei, tema și dificultatea. Pentru versiunea următoare, cel mai bun dataset ar fi o colecție extinsă din arhive oficiale + etichetare profesorală pentru competențe și erori conceptuale.

## Cele două servicii ML

### 1. Serviciu pe date structurate

- **Input:** `Tema_norm`, `Domeniu`, `Sursa_type`, `Itemul`, anul sursei, lungimi, număr de simboluri matematice, indicatori de procente/geometrie/ecuații/radicali/funcții/context real.
- **Output:** `Dificultate_group`: `1 - bază`, `2 - mediu`, `3 - consolidare`, `4 - avansat`.
- **Model:** `RandomForestClassifier` în pipeline cu imputare, scalare și OneHotEncoder.
- **Evaluare:** macro-F1 0.802, baseline macro-F1 0.163; CV 0.826 ± 0.094.
- **Alternative comparate:** GradientBoosting, LogisticRegression, XGBoost (opțional) — toate raportate; RandomForest ales pentru stabilitate CV + interpretabilitate.

### 2. Serviciu pe date nestructurate

- **Input:** enunț brut de problemă matematică.
- **Output:** domeniu curricular: Geometrie, Funcții, Ecuații/Inecuații/Sisteme, Mulțimi numerice etc.
- **Model:** TF-IDF + ComplementNB (`sublinear_tf=True`).
- **Evaluare:** macro-F1 0.974, baseline macro-F1 0.078; CV 0.993 ± 0.005.
- **Alternative comparate:** LinearSVC, LogisticRegression pe același TF-IDF — competitive, raportate transparent.

Cele două servicii sunt complementare: modelul text etichetează probleme noi, iar modelul structurat controlează progresia dificultății și recomandarea adaptivă.

## Rulare rapidă

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplicația folosește modelele deja salvate în `/models`. Dacă lipsește vreun artifact, app-ul va cere explicit să rulezi `python -m src.train_models` în loc să reantreneze în background.

Pentru reantrenare:

```bash
python -m src.train_models
```

Pentru regenerarea vizualizărilor EDA (heatmap corelații, word clouds, distribuții):

```bash
python -m src.eda
```

### Instalare dezvoltare

```bash
pip install -r requirements-dev.txt
```

### Testare

```bash
pytest
```

### API REST (opțional) — testare programatică a celor două servicii

Pe lângă interfața Streamlit, cele două servicii ML pot fi testate programatic
printr-un API FastAPI cu documentație Swagger:

```bash
pip install -r requirements.txt -r requirements-api.txt
uvicorn api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

Endpoint-uri: `POST /predict/domain` (text → domeniu), `POST /predict/difficulty`
(metadate → dificultate), `GET /health`, `GET /schema`.

## Structura proiectului

```text
app.py                              # UI Streamlit
src/app_helpers.py                  # validare active ML, inițializare session_state
src/data_prep.py                    # curățare, normalizare, feature engineering
src/train_models.py                 # antrenare + baseline + GridSearchCV + evaluare
src/model_utils.py                  # încărcare modele + inferență
src/pedagogical_engine.py           # indicii, mastery update, recomandări
tests/                              # teste unitare pentru logica principală
data/raw/                           # fișierele .xlsx furnizate
data/processed/exercises_processed.csv
models/structured_difficulty_model.joblib
models/unstructured_domain_model.joblib
models/evaluation_report.json
docs/competition_QA.md              # răspunsuri pregătite pentru juriu
``` 

## Ce demonstrează aplicația

1. Elevul alege o problemă și introduce un răspuns.
2. Modelul text prezice domeniul curricular din enunț.
3. Modelul structurat prezice dificultatea.
4. Motorul pedagogic oferă indiciu gradual și întrebare de conștientizare.
5. Sistemul actualizează stăpânirea estimată și recomandă următorul exercițiu.
6. Taburile de evaluare arată baseline, metrici, tuning, confuzii și erori concrete.

## Lecții din proiectul vechi / greșeli evitate

Versiunea veche DidactAI a fost penalizată deoarece „serviciile ML” erau euristici hardcodate, metricele erau constante fabricate, nu exista split, baseline, tuning sau analiză de erori, iar etica era absentă. Acest proiect remediază direct acele probleme: modelele sunt antrenate, metricile sunt calculate în `evaluation_report.json`, iar aplicația folosește efectiv ambele servicii.

## Limitări oneste

- Componenta neurală (TensorFlow) pentru starea elevului este **opțională și experimentală, antrenată pe date sintetice** — acuratețea ei reflectă datele generate, nu performanță reală. Cele două servicii ML principale (dificultate, domeniu) sunt validate pe date reale.
- Nu avem încă istoric real de elevi, deci knowledge tracing-ul din demo este o actualizare transparentă, nu model secvențial LSTM.
- Verificarea răspunsului este un checker simplu, nu un evaluator simbolic complet.
- Datasetul este mic și dezechilibrat; de aceea raportăm macro-F1 și balanced accuracy, nu doar accuracy.
- Augmentarea introduce near-duplicate; le-am **auditat** (29% dintre itemii de test au un near-duplicate în train) și am re-evaluat group-aware (`StratifiedGroupKFold`, 978 clustere) — macro-F1 a rămas 0.974, deci scorul nu e artefact de leakage. Vezi `docs/model_report.md §4.2`.
- Limitarea „dataset mic” este caracterizată prin curbe de învățare (`assets/learning_curve_*.png`), nu doar admisă.
- Problemele cu imagini/diagrame lipsă pot fi clasificate incorect.

## Etică și impact

**Confidențialitate.** Datasetul conține exerciții, nu date personale. Demo-ul nu persistă date personale; în producție am salva doar un profil anonimizat.

**Utilizare responsabilă.** Sistemul nu oferă soluția completă implicit și comunică faptul că predicțiile sunt suport educațional, nu verdict final. Indiciile sunt graduale, iar întrebările sunt metacognitive.

**Bias în date (discutat explicit).** Datasetul are mai multe surse de bias pe care le recunoaștem:
- **Sursă concentrată:** exercițiile provin dintr-o bancă de examen românească/moldovenească, deci notația, stilul și tipul de problemă reflectă acea tradiție curriculară, nu varietatea internațională.
- **Nivel specific:** acoperă gimnaziu/liceu de bază; un model antrenat aici nu generalizează la matematică de nivel superior.
- **Dezechilibru de clase:** clasa `2 - mediu` și domeniul „Ecuații/Inecuații/Sisteme” domină, iar `4 - avansat` are foarte puține exemple (15). De aceea raportăm macro-F1 și balanced accuracy și folosim `class_weight='balanced'`, dar predicțiile pe clasele rare rămân mai puțin sigure.
- **Augmentare:** parafrazările pot introduce un stil artificial repetitiv (vezi limitarea din `docs/model_report.md`).

**Impact negativ potențial (și mitigări).**
- *Dependență de indicii:* elevul poate cere indicii în loc să gândească. Mitigare: indicii graduale, întrebări metacognitive înainte de indiciu, soluția completă nu apare automat.
- *Dificultate greșit estimată → frustrare sau plictiseală:* dacă modelul supra/subestimează dificultatea, traseul adaptiv poate demotiva. Mitigare: recomandarea combină predicția ML cu mastery-ul estimat și cu reguli pedagogice, nu se bazează pe un singur scor; comunicăm că este o estimare.
- *Supra-încredere într-un model MVP:* metricile mari pot crea impresia de infailibilitate. Mitigare: afișăm baseline, erori reprezentative și limitări direct în aplicație (tabul „Evaluare & EDA”).

**Transparență.** Metricile sunt calculate, nu fabricate (lecție din versiunea veche), și pot fi regenerate de oricine cu `python -m src.train_models`.
