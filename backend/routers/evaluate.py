
from fastapi import APIRouter, Body
from backend.services.evaluation import evaluate_feature_csv
router = APIRouter()
@router.post('/evaluate')
def evaluate(payload: dict = Body(...)): return evaluate_feature_csv(payload['features_csv'])
