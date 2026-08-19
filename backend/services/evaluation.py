
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from backend.models.random_forest_model import load_rf
from backend.utils.json_utils import sanitize_for_json, sanitize_df
IGNORE = {'window_index','start','end','label'}
def evaluate_feature_csv(csv_path):
    df = sanitize_df(pd.read_csv(csv_path))
    if 'label' not in df.columns: return {'ok': False, 'reason': 'No label column in feature CSV'}
    X = df[[c for c in df.columns if c not in IGNORE and pd.api.types.is_numeric_dtype(df[c])]].fillna(0.0); y = df['label'].astype(str); rf = load_rf()
    if rf is None: return {'ok': False, 'reason': 'RandomForest model not trained yet'}
    preds = rf.predict(X)
    return sanitize_for_json({'ok': True, 'classification_report': classification_report(y, preds, output_dict=True), 'confusion_matrix': confusion_matrix(y, preds).tolist(), 'labels': sorted(y.unique().tolist())})
