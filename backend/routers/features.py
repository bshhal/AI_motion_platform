
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.utils.file_utils import save_upload_bytes, allowed_motion_file
from backend.services.zara_adapter import build_feature_dataframe
from backend.utils.json_utils import sanitize_for_json
router = APIRouter()
@router.post('/extract_features')
async def extract_features(file: UploadFile = File(...)):
    if not allowed_motion_file(file.filename): raise HTTPException(status_code=400, detail='Upload a motion CSV/TXT only, not a knowledge file.')
    path = save_upload_bytes(file.filename, await file.read()); raw_df, feat_df, fs, out = build_feature_dataframe(path)
    return sanitize_for_json({'sampling_rate_hz': fs, 'rows': len(raw_df), 'windows': len(feat_df), 'features_csv': str(out), 'preview': feat_df.head(10).to_dict(orient='records')})
