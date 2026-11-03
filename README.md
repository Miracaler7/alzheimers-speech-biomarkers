# Early Alzheimer's Detection via Acoustic Biomarkers & Multimodal Validation

An end-to-end clinical machine learning framework detecting cognitive decline and dementia using speech patterns from the Pitt Corpus.

## Architecture
- **Acoustic Profiling:** Extracts 1,008 features per subject (240 handcrafted acoustic/prosodic + 768 Wav2Vec representations).
- **Champion Classifier:** Soft Voting Ensemble combining Logistic Regression, Random Forest (200 estimators), and XGBoost (150 estimators).
- **Validation Engine:** Algorithmic verification evaluating:
  - $FQF$ (Feature Quality Factor)
  - $RCS$ (Risk Confidence Score: $P_{AD} \times FQF$)
  - $SRI$ (Severity Risk Index: $RCS \times 100$)
- **Explainability:** SHAP TreeExplainer attribution mapping top decisive vocal disfluency biomarkers.

## Validation Concordance
- **Model Test Recall:** ~85.5%
- **Mathematical Formula Concordance:** ~97.3%
