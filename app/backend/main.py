from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.database import create_db
from backend.routers.alerts import router as alerts_router
from backend.routers import auth as auth_router
from backend.routers.batches import router as batches_router
from backend.routers.dashboard import router as dashboard_router
from backend.routers.notifications import router as notifications_router
from backend.routers.profile import router as profile_router
from backend.routers.segments import router as segments_router
from backend.routers.videos import router as videos_router


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
create_db()

app = FastAPI(title="Human Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos_router)
app.include_router(auth_router.router)
app.include_router(batches_router)
app.include_router(segments_router)
app.include_router(dashboard_router)
app.include_router(profile_router)
app.include_router(alerts_router)
app.include_router(notifications_router)


@app.exception_handler(StarletteHTTPException)
def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "data": None, "message": str(exc.detail)},
    )


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"success": False, "data": None, "message": "Invalid request"},
    )


app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
