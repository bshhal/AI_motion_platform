
import math
import numpy as np
import pandas as pd

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return sanitize_for_json(obj.replace([np.inf, -np.inf], 0.0).fillna(0.0).to_dict(orient='records'))
    if isinstance(obj, pd.Series):
        return sanitize_for_json(obj.replace([np.inf, -np.inf], 0.0).fillna(0.0).to_dict())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, complex):
        obj = obj.real
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
    return obj

def sanitize_df(df):
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
