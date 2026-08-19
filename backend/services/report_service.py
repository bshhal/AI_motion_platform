
from backend.utils.json_utils import sanitize_for_json
def build_report(final_reasoning, raw_df, feat_df, docs, neighbors, anomalies):
    return sanitize_for_json({'final_prediction': final_reasoning, 'raw_rows': len(raw_df), 'window_count': len(feat_df), 'retrieved_docs': docs, 'retrieved_reference_windows': neighbors, 'anomalies': anomalies})
