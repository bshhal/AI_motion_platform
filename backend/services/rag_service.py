
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.utils.config import KNOWLEDGE_DIR
from backend.rag.vector_store import build_motion_store, query_motion_store

def _load_doc_chunks():
    chunks=[]
    for p in KNOWLEDGE_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in {'.txt','.md','.json'}:
            chunks.append({'source': p.name, 'text': p.read_text(encoding='utf-8', errors='ignore')})
    return chunks

def retrieve_docs(query, top_k=3):
    chunks = _load_doc_chunks();
    if not chunks: return []
    texts = [c['text'] for c in chunks]; vec = TfidfVectorizer(stop_words='english'); X = vec.fit_transform(texts + [query]); sims = cosine_similarity(X[-1], X[:-1]).ravel(); order = sims.argsort()[::-1][:top_k]
    return [{'source': chunks[i]['source'], 'score': float(sims[i]), 'text': chunks[i]['text'][:1200]} for i in order]

def build_motion_reference_store(feat_df): return build_motion_store(feat_df)
def retrieve_motion_neighbors(feat_df, k=5):
    numeric = feat_df.select_dtypes(include=['number']).replace([float('inf'),-float('inf')],0.0).fillna(0.0)
    return query_motion_store(numeric, k=k)
