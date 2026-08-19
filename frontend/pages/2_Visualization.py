
import streamlit as st, base64
st.title('Visualization')
res = st.session_state.get('last_result')
if not res: st.info('Run analysis first from Upload & Analyze page.')
else:
    plots = res.get('plots', {})
    if 'raw_axes' in plots: st.image(base64.b64decode(plots['raw_axes']), caption='Raw Motion Axes')
    if 'magnitude' in plots: st.image(base64.b64decode(plots['magnitude']), caption='Magnitude')
    st.json(res.get('report', {}))
