
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.utils.file_utils import save_upload_bytes, allowed_motion_file
from backend.services.pipeline import run_pipeline
from backend.utils.json_utils import sanitize_for_json
router = APIRouter()
@router.post('/analyze')
async def analyze(file: UploadFile = File(...), window_size: int = Form(100), stride: int = Form(50)):
    if not allowed_motion_file(file.filename): raise HTTPException(status_code=400, detail='Upload a motion CSV/TXT only, not a knowledge file.')
    path = save_upload_bytes(file.filename, await file.read())
    try:
        result = run_pipeline(path, window_size=window_size, stride=stride)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return sanitize_for_json(result)
