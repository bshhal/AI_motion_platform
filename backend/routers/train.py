
from fastapi import APIRouter, Body
from backend.services.trainer import train_from_feature_csv
router = APIRouter()
@router.post('/train')
def train(payload: dict = Body(...)): return train_from_feature_csv(payload['features_csv'])
