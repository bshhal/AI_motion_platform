
import io, base64
import matplotlib.pyplot as plt

def fig_to_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=140, bbox_inches='tight'); plt.close(fig); buf.seek(0); return base64.b64encode(buf.read()).decode('utf-8')

def make_motion_plots(raw_df):
    plots={}; fig, ax = plt.subplots(figsize=(10,4))
    for c in ['x','y','z']:
        if c in raw_df.columns: ax.plot(raw_df[c].head(2000).values, label=c)
    ax.set_title('Raw Motion Axes'); ax.legend(); plots['raw_axes'] = fig_to_b64(fig)
    if 'magnitude' in raw_df.columns:
        fig, ax = plt.subplots(figsize=(10,4)); ax.plot(raw_df['magnitude'].head(2000).values); ax.set_title('Magnitude'); plots['magnitude'] = fig_to_b64(fig)
    return plots
