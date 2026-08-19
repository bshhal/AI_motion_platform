
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers.health import router as health_router
from backend.routers.analyze import router as analyze_router
from backend.routers.features import router as features_router
from backend.routers.train import router as train_router
from backend.routers.evaluate import router as evaluate_router
from backend.routers.rag import router as rag_router
from backend.routers.reports import router as reports_router
app = FastAPI(title='Motion Sensor AI Data Scientist Platform')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.include_router(health_router); app.include_router(analyze_router); app.include_router(features_router); app.include_router(train_router); app.include_router(evaluate_router); app.include_router(rag_router); app.include_router(reports_router)
