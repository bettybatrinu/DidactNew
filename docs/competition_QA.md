# Didact AI - răspunsuri pregătite pentru juriu

## 1. Problema și relevanța

**Ce problemă concretă rezolvăm?**  
Rezolvăm lipsa de ghidare personalizată la matematică. Elevul primește de obicei un răspuns corect/greșit, nu un traseu adaptat la ce concepte stăpânește, unde se blochează și cât ajutor cere.

**Cine este utilizatorul final?**  
Elevul de gimnaziu/liceu care exersează matematică, iar beneficiar indirect este profesorul, care poate vedea ce teme sunt fragile și ce tipuri de intervenții ajută.

**De ce este important în practică?**  
Matematica cere pași, justificare și transfer. Un sistem care dă imediat soluția poate crea dependență; Didact AI oferă indicii graduale și întrebări de conștientizare.

**Ce obiectiv măsurabil urmărim?**  
Ținte pre-înregistrate (stabilite înainte de antrenarea finală): O1 — dificultate macro-F1 ≥ 0.60; O2 — domeniu macro-F1 ≥ 0.80; O3 — depășire clară a baseline-ului; O4 — raportarea stabilității CV (mean ± std). Plus un KPI educațional: rata de îmbunătățire a mastery-ului (Δmastery per exercițiu corect fără indicii) și raportul indicii/răspuns-corect, măsurate din `student_interactions.csv`.

**Cum știm că proiectul a avut succes?**  
Toate țintele pre-înregistrate sunt atinse: dificultate macro-F1 **0.802** (țintă ≥0.60), domeniu macro-F1 **0.974** (țintă ≥0.80), ambele peste baseline (0.16→0.80; 0.08→0.97), CV raportat cu deviație standard. Aplicația e funcțională, ambele modele integrate, metrici calculate reproductibil, EDA și limitări afișate.

## 2. Arhitectura soluției ML

**Care este serviciul ML pe date structurate?**  
Serviciul `structured_difficulty_model.joblib`: prezice dificultatea exercițiului pe baza metadatelor și feature-urilor tabelare.

**Care este serviciul ML pe date nestructurate?**  
Serviciul `unstructured_domain_model.joblib`: clasifică domeniul curricular din textul brut al problemei.

**Ce input și output are fiecare serviciu?**  
Structurat: input = tema normalizată, domeniu, tip sursă, item, lungimi, număr de simboluri, indicatori precum procente/geometrie/ecuație; output = `1 - bază`, `2 - mediu`, `3 - consolidare`, `4 - avansat`.  
Nestructurat: input = enunțul problemei ca text liber; output = domeniu curricular: Geometrie, Funcții, Ecuații/Inecuații/Sisteme, Mulțimi numerice etc.

**Sunt ambele implementate și demonstrabile?**  
Da. În Streamlit, tabul „Cele 2 servicii ML” permite inferență separată pentru fiecare, iar tabul „Tutor demo” le combină în același flux.

**Unde apar cele două servicii în proiect?**  
Modelele sunt în `/models`, antrenarea în `src/train_models.py`, preprocesarea în `src/data_prep.py`, inferența în `src/model_utils.py`, integrarea în `app.py`.

**De ce nu era suficient un singur serviciu?**  
Textul rezolvă etichetarea problemelor noi, dar nu controlează progresia dificultății. Modelul structurat estimează dificultatea, dar are nevoie de etichete/metadate. Împreună permit: problemă nouă -> domeniu -> dificultate -> traseu adaptiv.

**Ce pierdem dacă eliminăm unul?**  
Fără text, nu putem încadra automat exerciții noi. Fără structurat, recomandarea adaptivă nu mai știe cât de greu este exercițiul.

## 3. Date și preprocesare structurate

**De unde provin datele structurate?**  
Din workbook-ul furnizat `Exercises_CORRECTED (2).xlsx`, cu exerciții de matematică, pași de rezolvare, răspunsuri, dificultate și temă. Am folosit și `Domenii, categorii.xlsx` ca ghid curricular pentru maparea temelor în domenii.

**Ce reprezintă variabilele principale?**  
`Tema_norm`, `Domeniu`, `Sursa_type`, `Itemul`, `Sursa_year`, `problem_words`, `steps_chars`, `n_digits`, `n_math_symbols`, `has_percent`, `has_geometry_word`, `has_equation_word`, `has_radical`, `has_function_word`, `has_real_life_context`.

**Care este targetul?**  
`Dificultate_group`: dificultatea normalizată în patru clase: bază, mediu, consolidare, avansat.

**De ce sunt potrivite datele?**  
Pentru un tutor adaptiv avem nevoie exact de conținut matematic, etichete tematice și dificultate. Setul brut are 489 exerciții (477 cu temă și dificultate), augmentat la 1344 rânduri pentru antrenare.

**Limitări?**  
Setul este mic pentru producție, are 51 duplicate exacte și 12 rânduri fără temă/dificultate. Etichetele tematice inițiale erau zgomotoase (`Functii` vs `Funcții`, `Ecuatii` vs `Ecuații`), deci le-am normalizat.

**Cum am tratat lipsurile/anomaliile?**  
Am filtrat targeturile necunoscute la antrenare, am imputat numeric cu mediană și categoric cu cea mai frecventă valoare în pipeline, am grupat dificultățile rare 4/5/6 în clasa `4 - avansat`.

**Cum am tratat categoricele și numericele?**  
Categorice: `SimpleImputer` + `OneHotEncoder(handle_unknown='ignore')`. Numerice: `SimpleImputer(strategy='median')` + `StandardScaler`. Model final: `RandomForestClassifier`.

**Ce am observat în EDA?**  
Distribuția este dezechilibrată: clasa `2 - mediu` este dominantă, iar unele domenii precum geometria sunt mai frecvente decât altele. De aceea folosim balanced accuracy și macro-F1, nu doar accuracy.

**Ce decizie de modelare a influențat EDA?**  
Am folosit macro-F1 la GridSearchCV, am grupat dificultățile rare, am folosit split stratificat și am raportat baseline-ul majoritar.

## 4. Model structurat

**Ce model am ales și de ce?**  
RandomForestClassifier într-un pipeline scikit-learn. Este potrivit pentru amestec de features numerice și categorice, poate modela interacțiuni neliniare și este robust pe dataseturi mici/medii.

**Alternative analizate?**  
Da, comparate experimental cu aceeași preprocesare și aceeași validare 5-fold: RandomForest (CV 0.815±0.020), GradientBoosting (0.812±0.102), LogisticRegression (0.750±0.055) și XGBoost opțional (0.767±0.116). Plus baseline `DummyClassifier`. Am ales RandomForest pentru stabilitatea CV (cea mai mică deviație între candidații robusti) și interpretabilitate; alternativele sunt competitive pe test și raportate transparent în `evaluation_report.json` → `model_comparison`.

**Putem demonstra inferență?**  
Da, în tabul „Cele 2 servicii ML” se introduce o problemă și metadate, iar modelul returnează dificultatea estimată și probabilitățile.

**Cum justificăm complexitatea?**  
Nu folosim deep learning pentru date structurate mici. RandomForest este suficient de puternic, rapid, explicabil la nivel de feature importance și ușor de rulat local.

## 5. Evaluare și robustețe structurate

**Metrici folosite?**  
Accuracy, balanced accuracy, macro-F1, weighted-F1, classification report și confusion matrix.

**Baseline și depășire?**  
Baseline macro-F1: 0.163. Model final macro-F1: 0.802. Baseline accuracy: 0.483. Model accuracy: 0.929. CV macro-F1: 0.826 ± 0.094 (raportăm media ± deviația standard pe 5 folduri, nu un singur split).

**De ce split i.i.d. și nu temporal?**  
Avem `Sursa_year`, deci un split temporal e posibil tehnic. Am ales split aleator stratificat pentru că exercițiile sunt itemi independenți, nu o serie temporală cu autocorelație. Un split temporal ar fi relevant pentru predicția evoluției unui elev în timp (knowledge tracing secvențial), o extensie viitoare, nu obiectivul MVP. Menționăm explicit alternativa ca să nu tratăm tăcut datele ca i.i.d.

**Tuning?**  
Da, `GridSearchCV` pe `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`.

**Validare robustă?**  
Da, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` pe train și test holdout stratificat.

**Anti-overfitting?**  
Limităm `max_depth`, testăm `min_samples_leaf`, folosim CV și holdout separat, iar pipeline-ul evită leakage deoarece preprocessing-ul este fit doar pe train în CV.

## 6. Date nestructurate

**Tip de date nestructurate?**  
Text: enunțuri matematice în limba română, cu notație matematică.

**De unde provin?**  
Din aceeași bancă de exerciții furnizată. Textul problemei este coloana `Problema`.

**Cum sunt etichetate?**  
Fiecare problemă are `Tema`, normalizată în `Tema_norm` și apoi mapată în `Domeniu`.

**Care este targetul?**  
`Domeniu` curricular.

**Preprocesare?**  
Normalizare diacritice/case în vectorizator, tokenizare care păstrează simboluri matematice, TF-IDF cu unigrams/bigrams, eliminare duplicate exacte pentru reducerea leakage-ului.

**Analiză exploratorie?**  
Am analizat distribuția domeniilor și lungimea problemelor. Lungimea mediană este 13 cuvinte; există probleme foarte scurte, ceea ce poate reduce siguranța modelului.

**Concluzie EDA care a influențat modelul?**  
Pentru text scurt și dataset mic, un model TF-IDF + ComplementNB este mai potrivit și mai robust decât un transformer greu de justificat/rulat local.

## 7. Model și robustețe nestructurate

**Model ales și alternative comparate?**  
TF-IDF + ComplementNB (potrivit pentru text dezechilibrat). L-am comparat cu LinearSVC și LogisticRegression pe **același** TF-IDF, cu validare 5-fold: ComplementNB CV 0.993±0.005, LinearSVC 0.992±0.005, LogisticRegression 0.990±0.005. Toate sunt raportate în `model_comparison`; ComplementNB rămâne principal pentru viteză și simplitate, iar comparația confirmă că reprezentarea TF-IDF e puternică indiferent de clasificator. Am adăugat și `sublinear_tf=True` ca tehnică anti-overfitting specifică NLP (selectată de GridSearch).

**Am folosit transfer learning/model preantrenat?**  
Nu în MVP. Am ales un model clasic, reproductibil, rapid, care poate rula offline. Dacă avem timp, putem compara cu embeddings Romanian/RoBERT ca extensie, dar fără să riscăm demo-ul.

**Input/output exact?**  
Input: enunț brut. Output: domeniu curricular probabilistic.

**Inferență live?**  
Da, în tabul „Cele 2 servicii ML”.

**Metrici?**  
Accuracy, balanced accuracy, macro-F1, weighted-F1, confusion matrix, erori concrete.

**Baseline?**  
Baseline macro-F1: 0.078. Model final macro-F1: 0.974. Baseline accuracy: 0.305. Model accuracy: 0.982. CV macro-F1: 0.993 ± 0.005.

**Atenție onestă la scorul mare — auditat, nu doar semnalat:** am verificat leakage-ul din augmentare. În split naiv, 29% dintre itemii de test au un near-duplicate în train. Am re-evaluat **group-aware** (978 clustere de near-duplicate, `StratifiedGroupKFold`): macro-F1 a rămas **0.974** (CV 0.986 ± 0.009). Concluzie: scorul nu este artefact de leakage — taskul e genuin separabil (vocabular determinist pe domenii). Auditul complet e în `evaluation_report.json → leakage_audit`.

**Cum caracterizați limitarea „dataset mic”?**  
Prin curbe de învățare (`assets/learning_curve_*.png`): la text, CV macro-F1 se aplatizează ~0.99 (datasetul e suficient pentru acest task); la structurat, CV se aplatizează ~0.80 cu un gap train/CV (~0.96 vs ~0.80), semnătura unui set mic pentru un task mai greu — aici mai multe date reale ar ajuta cel mai mult. Cuantificăm limitarea, nu doar o admitem.

**Validare și anti-overfitting?**  
Split stratificat, GridSearchCV cu StratifiedKFold, limitare `max_features`, `min_df`, comparație cu baseline și eliminare duplicate exacte.

**Dovadă că generalizează?**  
Raportăm performanța pe test holdout care nu a fost folosit la tuning, plus exemple de erori.

## 8. Protocol de evaluare și analiză critică

**Cum am împărțit datele?**  
Train/test stratificat 78/22, apoi GridSearchCV cu 5-fold stratificat pe train.

**Cum prevenim leakage-ul?**  
Pipeline scikit-learn fit-uit în CV, nu preprocesăm pe tot datasetul înainte de split. Pentru text, eliminăm duplicate exacte înainte de split.

**De ce aceste metrici?**  
Accuracy singură poate ascunde dezechilibrul. Macro-F1 și balanced accuracy tratează mai corect clasele rare.

**Cum tratăm dezechilibrul?**  
Metrici macro, `class_weight='balanced'` la RandomForest și ComplementNB pentru text, plus analiză de distribuție.

**Erori frecvente?**  
Modelul text confundă teme apropiate: ecuații vs funcții când enunțul conține `f(x)` și rezolvare de ecuații. Modelul structurat confundă dificultăți vecine, mai ales `2 - mediu` vs `3 - consolidare`.

**Limitări principale?**  
Dataset mic, etichete tematice normalizate automat, lipsa istoricului real al elevilor, lipsa unui evaluator simbolic complet.

**Scenarii nesigure?**  
Probleme cu imagini lipsă, enunțuri foarte scurte, teme rare, formule ambigue sau probleme care cer diagramă.

**Ce am îmbunătăți prima dată?**  
Colectare de interacțiuni reale anonimizate, etichetare profesorală pentru erori conceptuale, evaluator simbolic al pașilor, comparație cu embeddings preantrenate.

## 9. Etică și impact

**Poate modelul introduce bias? (detaliat)**  
Da, și îl discutăm explicit pe surse:
- *Sursă concentrată:* exercițiile provin dintr-o singură bancă de examen românească/moldovenească, deci notația, stilul și tipurile de probleme reflectă acea tradiție curriculară, nu varietatea internațională. Un model antrenat aici va fi mai sigur pe stilul acelei surse.
- *Nivel specific:* datasetul acoperă gimnaziu/liceu de bază; nu generalizează la matematică de nivel superior.
- *Dezechilibru de clase:* `2 - mediu` și domeniul „Ecuații/Inecuații/Sisteme” domină, iar `4 - avansat` are doar 15 exemple. Predicțiile pe clasele rare sunt mai puțin sigure. Mitigare: macro-F1 + balanced accuracy + `class_weight='balanced'`.
- *Augmentare:* parafrazările pot introduce un stil artificial repetitiv și pot umfla scorul text (vezi nota GroupKFold din `model_report.md`).

**Ce impact negativ ar putea avea sistemul și cum îl atenuăm?**  
- *Dependență de indicii:* elevul cere indicii în loc să gândească. Mitigare: indicii graduale, întrebare metacognitivă înainte de indiciu, soluția completă nu apare automat.
- *Dificultate greșit estimată → frustrare sau plictiseală:* o estimare greșită poate demotiva elevul. Mitigare: recomandarea combină predicția ML cu mastery-ul și cu reguli pedagogice, nu se bazează pe un singur scor; comunicăm că este o estimare, nu un verdict.
- *Supra-încredere într-un model MVP:* metricile mari pot crea impresia de infailibilitate. Mitigare: afișăm baseline, erori reprezentative și limitări direct în tabul „Evaluare & EDA”.

**Cum protejăm datele sensibile?**  
Datasetul conține exerciții, nu date personale. Pentru elevi, MVP-ul nu persistă date personale; în producție am salva doar profil anonimizat.

**Riscuri de utilizare incorectă?**  
Elevul ar putea încerca să obțină soluția fără gândire. De aceea indiciile sunt graduale și soluția completă nu apare implicit.

**Măsuri de utilizare responsabilă?**  
Feedback descriptiv, întrebări metacognitive, transparență asupra limitărilor, fără metrici fabricate, fără promisiunea că modelul înlocuiește profesorul.

**Cum comunicăm limitele?**  
În tabul „Evaluare & EDA” și în README: modelul recomandă, nu decide definitiv; răspunsurile trebuie verificate de elev/profesor; componenta neurală este experimentală (date sintetice).

## 10. Aplicație, reproducibilitate și prezentare

**Ce aplicație am construit?**  
O aplicație Streamlit cu tutor demo, două servicii ML testabile separat, EDA, evaluare, Q&A și etică.

**Aplicația este funcțională la jurizare?**  
Da. Comanda: `pip install -r requirements.txt` și `streamlit run app.py`.

**Ce poate face utilizatorul?**  
Alege exerciții, răspunde, cere indicii, vede evaluarea, vede recomandarea următoarei probleme și testează separat modelele ML.

**Cum se observă valoarea practică?**  
Un exercițiu nou este încadrat curricular, dificultatea este estimată, iar elevul primește un traseu adaptiv în loc de soluție imediată.

**Poate juriul testa separat fiecare serviciu?**  
Da, tabul „Cele 2 servicii ML” are două formulare separate.

**Scenariu real cu ambele servicii?**  
Elevul introduce/alege o problemă. Modelul text prezice domeniul. Modelul structurat estimează dificultatea. Motorul pedagogic decide indiciu și următorul exercițiu.

**Organizarea proiectului?**  
`app.py` UI Streamlit (4 taburi, inclusiv „Evaluare & EDA”); `src/data_prep.py` curățare/feature engineering; `src/train_models.py` antrenare/evaluare/comparație de modele; `src/eda.py` vizualizări EDA; `src/model_utils.py` inferență; `src/pedagogical_engine.py` reguli tutor; `api.py` API REST FastAPI; `/models` artifacts; `/data` date; `/docs` documentație.

**Dependințe?**  
Python, Streamlit, pandas, numpy, scikit-learn, joblib, openpyxl. Opțional: TensorFlow (componenta neurală), matplotlib/seaborn/wordcloud (regenerare EDA), FastAPI/uvicorn (API REST).

**Pot fi testate serviciile programatic, nu doar din UI?**  
Da. `api.py` expune un API FastAPI cu Swagger: `pip install -r requirements-api.txt`, apoi `uvicorn api:app --port 8000` și `http://localhost:8000/docs`. Endpoint-uri: `POST /predict/domain`, `POST /predict/difficulty`, `GET /health`, `GET /schema`. Folosește exact aceleași modele ca aplicația Streamlit.

**Poate un evaluator reproduce?**  
Da. Modelele sunt salvate, dar pot fi reantrenate cu `python -m src.train_models`, iar vizualizările cu `python -m src.eda`. Raportul JSON și figurile sunt regenerate din date — nimic nu este hardcodat.
