
import streamlit as st, requests
BACKEND='http://127.0.0.1:8000'
st.title('RAG Explorer')
query = st.text_input('Ask motion/HAR question', 'walking vs jogging cadence and anomaly detection')
if st.button('Retrieve'): st.json(requests.post(f'{BACKEND}/rag/query', json={'query': query, 'top_k': 3}).json())
res = st.session_state.get('last_result')
if res:
    st.subheader('Last Run Retrieved Docs'); st.json(res.get('retrieved_docs', [])); st.subheader('Last Run Similar Windows'); st.json(res.get('retrieved_reference_windows', []))
