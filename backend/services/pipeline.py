
from pathlib import Path
from backend.services.zara_adapter import build_feature_dataframe
from backend.services.model_service import predict_with_models
from backend.services.anomaly_service import detect_anomalies, detect_fall, detect_sensor_drift, detect_unusual_gait, detect_unknown_pattern
from backend.services.rag_service import retrieve_docs, build_motion_reference_store, retrieve_motion_neighbors
from backend.services.visualization_service import make_motion_plots
from backend.services.reasoning_service import reason_over_evidence
from backend.services.report_service import build_report
from backend.utils.json_utils import sanitize_df

def _baseline_prediction(feat_df):
    pred = feat_df['label'].mode().iloc[0] if 'label' in feat_df.columns and not feat_df['label'].isna().all() else 'Unknown'
    return pred, 0.55

def run_pipeline(file_path: Path, window_size: int = 100, stride: int = 50):
    raw_df, feat_df, fs, feat_csv = build_feature_dataframe(file_path, window_size, stride)
    feat_df, anomaly_summary = detect_anomalies(feat_df)
    fall_summary = detect_fall(raw_df); drift_summary = detect_sensor_drift(raw_df); gait_summary = detect_unusual_gait(feat_df)
    build_motion_reference_store(feat_df); neighbors = retrieve_motion_neighbors(feat_df, k=5)
    model_preds = predict_with_models(feat_df)
    base_pred, base_conf = _baseline_prediction(feat_df)
    if 'random_forest' in model_preds: base_pred, base_conf = model_preds['random_forest']['majority_prediction'], 0.75
    docs = retrieve_docs(f'{base_pred} motion sensor activity anomaly gait fall drift', top_k=3)
    unknown_summary = detect_unknown_pattern({'confidence': base_conf}, anomaly_summary)
    evidence = {'base_prediction': base_pred, 'confidence': base_conf, 'sampling_rate_hz': fs, 'model_predictions': model_preds, 'anomaly_summary': {**anomaly_summary, **fall_summary, **drift_summary, **gait_summary, **unknown_summary}, 'retrieved_docs': docs, 'retrieved_reference_windows': neighbors, 'window_feature_summary': sanitize_df(feat_df.select_dtypes(include=['number'])).mean().to_dict()}
    final_reasoning = reason_over_evidence(evidence); plots = make_motion_plots(raw_df); report = build_report(final_reasoning, raw_df, feat_df, docs, neighbors, evidence['anomaly_summary'])
    return {'prediction': final_reasoning, 'sampling_rate_hz': fs, 'features_csv': str(feat_csv), 'model_predictions': model_preds, 'retrieved_docs': docs, 'retrieved_reference_windows': neighbors, 'anomaly_detection': evidence['anomaly_summary'], 'plots': plots, 'report': report}
