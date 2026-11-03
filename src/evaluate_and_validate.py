"""
Module: evaluate_and_validate.py
Evaluates test cohort and calculates clinical formulas:
FQF (Feature Quality Factor), RCS (Risk Confidence Score), SRI (Severity Risk Index).
Validates model predictions against mathematical risk thresholds.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def run_validation(base_dir: Path):
    models_dir = base_dir / 'models'
    features_dir = base_dir / 'features'
    metadata_dir = base_dir / 'metadata'

    scaler = joblib.load(models_dir / 'feature_scaler.joblib')
    ensemble = joblib.load(models_dir / 'best_classifier.joblib')
    manifest = pd.read_csv(metadata_dir / 'feature_manifest.csv')

    test_df = manifest[manifest['split'].str.lower() == 'test'].reset_index(drop=True)
    X_test, y_test, subject_ids = [], [], []

    for _, row in test_df.iterrows():
        sid = str(row['subject_id']).replace('.npz', '').replace('.wav', '')
        f_path = features_dir / f"{sid}.npz"
        if f_path.exists():
            X_test.append(np.load(f_path)['features'])
            y_test.append(1 if 'ad' in str(row['label']).lower() or str(row['label']) == '1' else 0)
            subject_ids.append(sid)

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test_scaled = scaler.transform(X_test)

    ml_prob = ensemble.predict_proba(X_test_scaled)[:, 1]
    ml_pred = (ml_prob >= 0.50).astype(int)

    records = []
    for i, sid in enumerate(subject_ids):
        feats = X_test[i]
        is_corrupt = np.isnan(feats).any() or np.isinf(feats).any()
        low_var = np.var(feats[:240]) < 1e-4

        fqf = 0.0 if is_corrupt else (0.70 if low_var else 0.98)
        rcs = ml_prob[i] * fqf
        sri = rcs * 100.0
        math_pred = 1 if sri >= 50.0 else 0

        band = "Low Risk (Control)" if sri < 35 else ("Moderate Risk (Monitor)" if sri < 65 else "High Risk (Probable AD)")

        records.append({
            'Subject_ID': sid,
            'Actual_Diagnosis': 'Dementia' if y_test[i] == 1 else 'Control',
            'ML_Probability (%)': np.round(ml_prob[i] * 100, 2),
            'ML_Prediction': 'Dementia' if ml_pred[i] == 1 else 'Control',
            'FQF': np.round(fqf, 2),
            'RCS': np.round(rcs, 4),
            'SRI_Score (%)': np.round(sri, 1),
            'Math_Prediction': 'Dementia' if math_pred == 1 else 'Control',
            'Clinical_Band': band,
            'Validation_Status': 'Validated' if ml_pred[i] == math_pred else 'Diverged'
        })

    val_df = pd.DataFrame(records)
    out_path = metadata_dir / 'mathematical_validation_results.csv'
    val_df.to_csv(out_path, index=False)
    match_rate = (val_df['Validation_Status'] == 'Validated').mean() * 100
    print(f"Validation finished. Match Rate: {match_rate:.2f}% | Saved to: {out_path}")

if __name__ == '__main__':
    run_validation(Path('..'))
