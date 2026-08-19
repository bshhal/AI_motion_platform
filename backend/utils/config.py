
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
UPLOADED_DIR = DATA_DIR / 'uploaded'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = DATA_DIR / 'models'
REPORTS_DIR = DATA_DIR / 'reports'
KNOWLEDGE_DIR = DATA_DIR / 'knowledge'
VECTOR_DIR = DATA_DIR / 'vector_store'
for p in [DATA_DIR, UPLOADED_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, KNOWLEDGE_DIR, VECTOR_DIR]:
    p.mkdir(parents=True, exist_ok=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
DEFAULT_WINDOW = 100
DEFAULT_STRIDE = 50
