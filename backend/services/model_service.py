
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from backend.models.random_forest_model import load_rf, train_rf
from backend.models.xgboost_model import load_xgb, train_xgb
from backend.utils.json_utils import sanitize_df
IGNORE = {'window_index','start','end','label'}
def prepare_xy(feat_df):
    feat_df = sanitize_df(feat_df)
    cols = [c for c in feat_df.columns if c not in IGNORE and pd.api.types.is_numeric_dtype(feat_df[c])]
    X = feat_df[cols].replace([np.inf,-np.inf],0.0).fillna(0.0); y = feat_df['label'].astype(str) if 'label' in feat_df.columns else None
    return X, y, cols
def train_models(feat_df):
    X, y, cols = prepare_xy(feat_df)
    if y is None: return {'trained': False, 'reason': 'No labels found in uploaded data'}
    le = LabelEncoder(); y_enc = le.fit_transform(y); train_rf(X, y); xgb = train_xgb(X, y_enc)
    return {'trained': True, 'classes': list(le.classes_), 'feature_count': len(cols), 'xgb_available': xgb is not None}
def predict_with_models(feat_df):
    X, y, cols = prepare_xy(feat_df); rf = load_rf(); xgb = load_xgb(); result = {}
    if rf is not None:
        preds = rf.predict(X); result['random_forest'] = {'window_predictions': preds.tolist(), 'majority_prediction': pd.Series(preds).mode().iloc[0]}
    if xgb is not None:
        preds = xgb.predict(X); result['xgboost'] = {'window_predictions': preds.tolist(), 'majority_prediction': int(pd.Series(preds).mode().iloc[0])}
    return result
