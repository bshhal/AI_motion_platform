
from pathlib import Path
import numpy as np, pandas as pd
from backend.services.preprocessing_service import smart_read_csv, normalize_motion_df, estimate_sampling_rate
from backend.utils.json_utils import sanitize_df
from backend.utils.config import PROCESSED_DIR
from zara_core.data_preprocess import split_sequences
from zara_core.get_feats import extract_features

def _sanitize_feature_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, complex): v = v.real
        try: v = float(v)
        except Exception: v = 0.0
        if np.isnan(v) or np.isinf(v): v = 0.0
        out[k] = v
    return out

def build_feature_dataframe(file_path: Path, window_size: int = 100, stride: int = 50):
    raw = smart_read_csv(file_path)
    df = normalize_motion_df(raw)
    fs = estimate_sampling_rate(df)
    seq = df[['x','y','z']].to_numpy(dtype=float)
    segments, labels = split_sequences(seq.tolist(), window_size, stride)
    rows = []
    for i, seg in enumerate(segments):
        sensor_data = {'T_acc': np.asarray(seg, dtype=float).T}
        feats = extract_features(sensor_data=sensor_data, fs=fs, channel_names=list(sensor_data.keys()))
        feats = _sanitize_feature_dict(feats)
        feats['window_index'] = i; feats['start'] = labels[i][0]; feats['end'] = labels[i][1]
        if 'activity_decoded' in df.columns:
            w = df.iloc[labels[i][0]:labels[i][1]+1]['activity_decoded']
            feats['label'] = str(w.mode().iloc[0]) if not w.empty else 'Unknown'
        rows.append(feats)
    feat_df = sanitize_df(pd.DataFrame(rows))
    out = PROCESSED_DIR / f"{file_path.stem}_zara_features.csv"
    feat_df.to_csv(out, index=False)
    return df, feat_df, fs, out
