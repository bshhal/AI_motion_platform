
import streamlit as st, requests
BACKEND='http://127.0.0.1:8000'
st.title('Model Comparison')
res = st.session_state.get('last_result')
if not res: st.info('Run analysis first.')
else:
    st.json(res.get('model_predictions', {}))
    if st.button('Train Models on Last Features CSV'): st.json(requests.post(f'{BACKEND}/train', json={'features_csv': res['features_csv']}, timeout=600).json())
    if st.button('Evaluate Models on Last Features CSV'): st.json(requests.post(f'{BACKEND}/evaluate', json={'features_csv': res['features_csv']}, timeout=600).json())
