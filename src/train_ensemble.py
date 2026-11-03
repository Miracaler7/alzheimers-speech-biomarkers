"""
Module: train_ensemble.py
Trains the Soft Voting Ensemble (Logistic Regression, Random Forest, XGBoost)
over 1,008 acoustic & Wav2Vec features and saves standardized artifacts.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

def train_pipeline(base_dir: Path):
    models_dir = base_dir / 'models'
    features_dir = base_dir / 'features'
    metadata_dir = base_dir / 'metadata'
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(metadata_dir / 'feature_manifest.csv')
    train_df = manifest[manifest['split'].str.lower() == 'train'].reset_index(drop=True)

    X_train, y_train = [], []
    for _, row in train_df.iterrows():
        sid = str(row['subject_id']).replace('.npz', '').replace('.wav', '')
        f_path = features_dir / f"{sid}.npz"
        if f_path.exists():
            X_train.append(np.load(f_path)['features'])
            y_train.append(1 if 'ad' in str(row['label']).lower() or str(row['label']) == '1' else 0)

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, models_dir / 'feature_scaler.joblib')

    # Ensemble Classifiers
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.05, random_state=42, eval_metric='logloss')

    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train)
    joblib.dump(ensemble, models_dir / 'best_classifier.joblib')
    print("Training complete. Artifacts saved to models/")

if __name__ == '__main__':
    train_pipeline(Path('..'))
