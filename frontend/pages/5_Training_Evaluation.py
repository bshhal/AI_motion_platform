
import streamlit as st, requests
BACKEND='http://127.0.0.1:8000'
st.title('Training & Evaluation')
features_csv = st.text_input('Feature CSV path')
col1,col2 = st.columns(2)
if col1.button('Train') and features_csv: st.json(requests.post(f'{BACKEND}/train', json={'features_csv': features_csv}, timeout=600).json())
if col2.button('Evaluate') and features_csv: st.json(requests.post(f'{BACKEND}/evaluate', json={'features_csv': features_csv}, timeout=600).json())
