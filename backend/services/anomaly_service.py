
import numpy as np
from sklearn.ensemble import IsolationForest
from backend.utils.json_utils import sanitize_df
IGNORE = {'window_index','start','end','label'}
def detect_anomalies(feat_df):
    feat_df = sanitize_df(feat_df); numeric = [c for c in feat_df.columns if c not in IGNORE and feat_df[c].dtype.kind in 'if']
    X = feat_df[numeric].replace([np.inf,-np.inf],0.0).fillna(0.0); iso = IsolationForest(random_state=42, contamination=0.05); labels = iso.fit_predict(X); scores = iso.decision_function(X)
    feat_df = feat_df.copy(); feat_df['anomaly_flag'] = (labels == -1).astype(int); feat_df['anomaly_score'] = -scores
    return feat_df, {'anomalous_windows': int(feat_df['anomaly_flag'].sum()), 'mean_anomaly_score': float(feat_df['anomaly_score'].mean())}
def detect_fall(raw_df):
    mag = raw_df['magnitude'].to_numpy() if 'magnitude' in raw_df.columns else np.array([])
    if mag.size == 0: return {'fall_detected': False, 'reason': 'No magnitude'}
    peak = float(np.max(mag)); stillness = float(np.std(mag[-max(50, len(mag)//20):])); return {'fall_detected': bool(peak > 18 and stillness < 1.2), 'peak_magnitude': peak, 'post_peak_stillness_std': stillness}
def detect_sensor_drift(raw_df):
    out = {}
    for axis in ['x','y','z']:
        if axis in raw_df.columns:
            arr = raw_df[axis].to_numpy(dtype=float); t = np.arange(len(arr), dtype=float); den = ((t-t.mean())**2).sum(); slope = (((t-t.mean())*(arr-arr.mean())).sum()/den) if den>0 else 0.0; out[f'{axis}_drift_slope'] = float(slope)
    out['sensor_drift_detected'] = bool(any(abs(v) > 0.01 for k,v in out.items() if k.endswith('_slope')))
    return out
def detect_unusual_gait(feat_df):
    cols = [c for c in feat_df.columns if 'fft_dom_freq' in c or 'rms' in c or 'vel_std' in c]
    if not cols: return {'unusual_gait_detected': False, 'reason': 'Missing gait features'}
    x = feat_df[cols].replace([np.inf,-np.inf],0.0).fillna(0.0); variability = float(x.std().mean()); return {'unusual_gait_detected': bool(variability > 5.0), 'gait_variability_score': variability}
def detect_unknown_pattern(prediction_summary, anomaly_summary):
    unknown = anomaly_summary.get('anomalous_windows',0) > 0 and prediction_summary.get('confidence',0) < 0.55
    return {'unknown_pattern_detected': bool(unknown), 'message': 'This pattern does not match known classes' if unknown else 'Pattern matched known classes'}
