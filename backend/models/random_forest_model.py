
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
MODEL_PATH = Path(__file__).resolve().parents[2] / 'data' / 'models' / 'rf_model.joblib'
def train_rf(X, y):
    model = RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=42)
    model.fit(X, y); joblib.dump(model, MODEL_PATH); return model
def load_rf():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
