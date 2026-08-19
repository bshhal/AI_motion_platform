
from pathlib import Path
from .config import UPLOADED_DIR

def save_upload_bytes(filename: str, content: bytes) -> Path:
    path = UPLOADED_DIR / filename
    with open(path, 'wb') as f:
        f.write(content)
    return path

def allowed_motion_file(filename: str) -> bool:
    lower = filename.lower()
    if lower in {'har_knowledge.txt', 'activity_pair_knowledge.json'}:
        return False
    return lower.endswith('.csv') or lower.endswith('.txt')
