
import streamlit as st, requests
BACKEND='http://127.0.0.1:8000'
st.title('Reports')
r = requests.get(f'{BACKEND}/reports/list')
if r.ok: st.json(r.json())
res = st.session_state.get('last_result')
if res: st.subheader('Last Run Final Report'); st.json(res.get('report', {}))
