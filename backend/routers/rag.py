
from fastapi import APIRouter, Body
from backend.services.rag_service import retrieve_docs
router = APIRouter()
@router.post('/rag/query')
def rag_query(payload: dict = Body(...)): return {'results': retrieve_docs(payload.get('query',''), top_k=int(payload.get('top_k',3)))}
