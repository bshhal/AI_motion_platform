
from fastapi import APIRouter
from backend.utils.config import REPORTS_DIR, PROCESSED_DIR
router = APIRouter()
@router.get('/reports/list')
def list_reports(): return {'processed_files': [p.name for p in PROCESSED_DIR.glob('*.csv')], 'report_files': [p.name for p in REPORTS_DIR.glob('*')]}
