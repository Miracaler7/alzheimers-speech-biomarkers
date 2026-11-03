"""
Module: explainability_shap.py
Generates TreeSHAP summary attribution plots across the 1,008 acoustic features.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shap

def run_shap(base_dir: Path):
    models_dir = base_dir / 'models'
    features_dir = base_dir / 'features'
    metadata_dir = base_dir / 'metadata'

    scaler = joblib.load(models_dir / 'feature_scaler.joblib')
    ensemble = joblib.load(models_dir / 'best_classifier.joblib')
    manifest = pd.read_csv(metadata_dir / 'feature_manifest.csv')

    test_df = manifest[manifest['split'].str.lower() == 'test'].reset_index(drop=True)
    X_test = []
    for _, row in test_df.iterrows():
        sid = str(row['subject_id']).replace('.npz', '').replace('.wav', '')
        f_path = features_dir / f"{sid}.npz"
        if f_path.exists():
            X_test.append(np.load(f_path)['features'])

    X_test_scaled = scaler.transform(np.array(X_test))
    xgb_estimator = ensemble.named_estimators_['xgb']
    
    explainer = shap.TreeExplainer(xgb_estimator)
    shap_vals = explainer.shap_values(X_test_scaled)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_test_scaled, max_display=20, show=False)
    plt.title("Top Acoustic Biomarker Contributions (SHAP TreeExplainer)")
    plt.tight_layout()
    plt.savefig(base_dir / 'shap_summary_plot.png', dpi=300)
    print("SHAP explanation plot generated and saved.")

if __name__ == '__main__':
    run_shap(Path('..'))
