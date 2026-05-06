# Robust cross-lingual fact checking — lightweight baseline lab

## Data

`synthetic_claims.csv` is a **tiny multilingual** illustration (EN/FA/TR) with 4 ordinal-ish labels (0–3). It is **not** a replacement for X-Fact / SciFact; it exists to show **evaluation + error analysis** mechanics on text.

## Quality

`analytics/quality/validate_claims.py` enforces schema + label range.

## Modeling

`modeling/tfidf_baseline.py` trains a **character n-gram TF–IDF + multinomial logistic** model (strong for noisy multilingual snippets without tokenizers), exports:

- `metrics.json`, `classification_report.txt`  
- `confusion_matrix.png`  
- `errors_by_lang.csv` + `error_sample.csv`

This complements your heavy XLM-R notebooks in the same repository.
