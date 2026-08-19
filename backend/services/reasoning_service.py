
import json
from backend.utils.config import OPENAI_API_KEY, OPENAI_MODEL
from zara_core.call_api import build_openai_client, call_gpt_with_retry
SYSTEM = """
You are a motion-sensor AI data scientist.
You must explain the result using model outputs, anomaly signals, retrieved HAR docs, and similar windows.
Return valid JSON with keys: predicted_activity, confidence, reason, anomaly_summary.
"""
def reason_over_evidence(payload):
    if not OPENAI_API_KEY:
        return {'predicted_activity': payload.get('base_prediction','Unknown'), 'confidence': payload.get('confidence',0.0), 'reason': 'LLM reasoning skipped because OPENAI_API_KEY is missing.', 'anomaly_summary': payload.get('anomaly_summary',{})}
    client = build_openai_client(OPENAI_API_KEY); prompt = json.dumps(payload, default=str)[:20000]; resp = call_gpt_with_retry(client, prompt, system_prompt=SYSTEM, model_name=OPENAI_MODEL, temp=0)
    if isinstance(resp, dict):
        return {'predicted_activity': payload.get('base_prediction','Unknown'), 'confidence': payload.get('confidence',0.0), 'reason': str(resp), 'anomaly_summary': payload.get('anomaly_summary',{})}
    try: return json.loads(resp)
    except Exception: return {'predicted_activity': payload.get('base_prediction','Unknown'), 'confidence': payload.get('confidence',0.0), 'reason': resp, 'anomaly_summary': payload.get('anomaly_summary',{})}
