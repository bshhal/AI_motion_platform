
import pandas as pd
from backend.models.random_forest_model import train_rf
from backend.models.xgboost_model import train_xgb
from backend.utils.json_utils import sanitize_df
IGNORE = {'window_index','start','end','label'}
def train_from_feature_csv(csv_path):
    df = sanitize_df(pd.read_csv(csv_path))
    if 'label' not in df.columns: return {'trained': False, 'reason': 'No label column in feature CSV'}
    X = df[[c for c in df.columns if c not in IGNORE and pd.api.types.is_numeric_dtype(df[c])]].fillna(0.0); y = df['label'].astype(str)
    train_rf(X, y); xgb = train_xgb(X, y); return {'trained': True, 'rows': len(df), 'feature_count': X.shape[1], 'xgb_available': xgb is not None}
