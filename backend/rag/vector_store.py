
from pathlib import Path
import pickle, numpy as np
from sklearn.neighbors import NearestNeighbors
from backend.utils.config import VECTOR_DIR
MOTION_STORE = VECTOR_DIR / 'motion_store.pkl'
def save_pickle(path, obj):
    with open(path, 'wb') as f: pickle.dump(obj, f)
def load_pickle(path, default):
    if path.exists():
        with open(path, 'rb') as f: return pickle.load(f)
    return default
def build_motion_store(feature_df, meta_cols=('window_index','start','end','label')):
    numeric = feature_df.select_dtypes(include=['number']).replace([np.inf,-np.inf],0.0).fillna(0.0)
    nn = NearestNeighbors(n_neighbors=min(10, len(numeric)), metric='cosine'); nn.fit(numeric.values)
    meta = feature_df[[c for c in meta_cols if c in feature_df.columns]].to_dict(orient='records'); save_pickle(MOTION_STORE, {'nn': nn, 'X': numeric.values, 'meta': meta, 'cols': list(numeric.columns)}); return {'stored_windows': len(meta), 'feature_dim': numeric.shape[1]}
def query_motion_store(query_df, k=5):
    store = load_pickle(MOTION_STORE, None)
    if store is None: return []
    cols = store['cols']; q = query_df.reindex(columns=cols, fill_value=0.0).replace([np.inf,-np.inf],0.0).fillna(0.0).values; dists, inds = store['nn'].kneighbors(q[:1], n_neighbors=min(k, len(store['meta']))); res=[]
    for d, i in zip(dists[0], inds[0]):
        item = dict(store['meta'][int(i)]); item['similarity'] = float(1.0 - d); res.append(item)
    return res
