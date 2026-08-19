
from pathlib import Path
import pandas as pd
import numpy as np
from backend.utils.json_utils import sanitize_df
ACTIVITY_MAP = {'A': 'Walking', 'B': 'Jogging', 'C': 'Stairs', 'D': 'Sitting', 'E': 'Standing'}
def smart_read_csv(path: Path) -> pd.DataFrame:
    if path.name.startswith(('body_acc_', 'body_gyro_', 'total_acc_')):
        raise ValueError(
            f'{path.name} looks like one UCI HAR axis file. Upload one file that contains all three axes as x,y,z columns instead.'
        )
    attempts = [
        {'sep': ',', 'header': 0},
        {'sep': ';', 'header': 0},
        {'sep': r'\s+', 'header': 0},
        {'sep': ',', 'header': None},
        {'sep': ';', 'header': None},
        {'sep': r'\s+', 'header': None},
    ]
    last = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] >= 3:
                return df
        except Exception as e:
            last = e
    detail = f': {last}' if last else ''
    raise ValueError(f'Could not parse {path.name}{detail}. Expected a CSV/TXT file with at least x, y, and z columns.')

def normalize_motion_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {'subject-id':'subject','subject':'subject','activity label':'activity','label':'activity','timestamp':'timestamp','x-accel':'x','y-accel':'y','z-accel':'z','acc_x':'x','acc_y':'y','acc_z':'z'}
    df = df.rename(columns=rename)
    if not all(c in df.columns for c in ['x','y','z']):
        if df.shape[1] >= 6:
            tmp = df.iloc[:, :6].copy(); tmp.columns = ['subject','activity','timestamp','x','y','z']; df = tmp
        elif df.shape[1] >= 3:
            tmp = df.iloc[:, -3:].copy(); tmp.columns = ['x','y','z']; df = tmp
        else:
            raise ValueError('Could not detect x/y/z columns')
    for c in ['x','y','z']:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(';', '', regex=False).str.strip(), errors='coerce')
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    if 'activity' in df.columns:
        df['activity'] = df['activity'].astype(str).str.strip(); df['activity_decoded'] = df['activity'].map(ACTIVITY_MAP).fillna(df['activity'])
    df = df.dropna(subset=['x','y','z']).reset_index(drop=True)
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    return sanitize_df(df)

def estimate_sampling_rate(df: pd.DataFrame, fallback_hz: float = 20.0) -> float:
    if 'timestamp' not in df.columns: return fallback_hz
    ts = pd.to_numeric(df['timestamp'], errors='coerce').dropna().to_numpy()
    if len(ts) < 5: return fallback_hz
    diffs = np.diff(ts); diffs = diffs[diffs > 0]
    if len(diffs) == 0: return fallback_hz
    med = float(np.median(diffs)); candidates = [1e9/med, 1e6/med, 1e3/med, 1.0/med]
    plausible = [c for c in candidates if 5 <= c <= 500]
    if plausible:
        common = [20,25,30,50,100,128,200]
        return min(plausible, key=lambda x: min(abs(x-c) for c in common))
    return fallback_hz
