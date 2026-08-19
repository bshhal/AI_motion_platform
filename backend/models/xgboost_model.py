
from pathlib import Path
import joblib
try:
    import xgboost as xgb
except Exception:
    xgb = None
MODEL_PATH = Path(__file__).resolve().parents[2] / 'data' / 'models' / 'xgb_model.joblib'
def train_xgb(X, y):
    if xgb is None: return None
    model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', random_state=42)
    model.fit(X, y); joblib.dump(model, MODEL_PATH); return model
def load_xgb():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
