# Lecții din jurizare și greșeli evitate

## Greșeli ale versiunii vechi DidactAI

- Nu avea modele ML antrenate: recomandarea era o distanță euristică între vectori.
- Pipeline-ul NLP era keyword matching, nu model pe text.
- Dashboard-ul afișa metrici statice/fabricate.
- Nu exista split de date, baseline, CV, tuning sau analiză de erori.
- Nu exista secțiune etică dedicată.
- README-ul explica ideea, dar nu demonstra tehnic criteriile.

## Cum repară acest MVP

- Modele reale salvate în `/models`.
- Antrenare reproductibilă în `src/train_models.py`.
- Baseline + GridSearchCV + StratifiedKFold + metrici reale.
- Rapoarte JSON și confusion matrices afișate în UI.
- Aplicație Streamlit care folosește efectiv ambele servicii.
- Q&A complet și secțiune de etică.
