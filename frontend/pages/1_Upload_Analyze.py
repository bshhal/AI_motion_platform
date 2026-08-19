
import streamlit as st, requests, base64
BACKEND='http://127.0.0.1:8000'
st.title('Upload & Analyze')
file = st.file_uploader('Upload motion CSV/TXT', type=['csv','txt'])
col1,col2=st.columns(2); window = col1.number_input('Window Size', min_value=20, max_value=1000, value=100); stride = col2.number_input('Stride', min_value=5, max_value=500, value=50)
if file and st.button('Run Full Analysis', use_container_width=True):
    files={'file': (file.name, file.getvalue())}; data={'window_size': int(window), 'stride': int(stride)}
    try:
        r = requests.post(f'{BACKEND}/analyze', files=files, data=data, timeout=600)
    except requests.exceptions.ConnectionError:
        st.error(f'Backend is not reachable at {BACKEND}. Start FastAPI on port 8000, then try again.')
        st.stop()
    if r.ok:
        res = r.json(); st.session_state['last_result'] = res; st.subheader('Prediction'); st.json(res['prediction']); st.subheader('Anomaly Detection'); st.json(res['anomaly_detection']); st.subheader('Model Predictions'); st.json(res['model_predictions']); st.subheader('Retrieved Docs'); st.json(res['retrieved_docs']); st.subheader('Retrieved Similar Windows'); st.json(res['retrieved_reference_windows']); st.subheader('Plots')
        if 'raw_axes' in res['plots']: st.image(base64.b64decode(res['plots']['raw_axes']), caption='Raw Axes')
        if 'magnitude' in res['plots']: st.image(base64.b64decode(res['plots']['magnitude']), caption='Magnitude')
        st.success(f"Feature CSV saved to: {res['features_csv']}")
    else:
        try:
            st.error(r.json().get('detail', r.text))
        except Exception:
            st.error(r.text)
