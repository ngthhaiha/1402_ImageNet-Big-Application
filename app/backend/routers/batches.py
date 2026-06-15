from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import get_db
from backend.models import Batch, User, Video
from backend.schemas import ApiResponse, BatchDetailRead, BatchRead, VideoRead


router = APIRouter(prefix="/api/batches", tags=["batches"])


def _api_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "data": None, "message": message},
    )


@router.get("/latest", response_model=ApiResponse[BatchDetailRead])
def get_latest_batch(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[BatchDetailRead] | JSONResponse:
    batch = db.query(Batch).order_by(Batch.created_at.desc(), Batch.id.desc()).first()
    if batch is None:
        return _api_error("No batch found", status_code=404)

    videos = (
        db.query(Video)
        .filter(Video.batch_id == batch.id)
        .order_by(Video.created_at.asc(), Video.id.asc())
        .all()
    )
    batch_data = BatchRead.model_validate(batch).model_dump()
    return ApiResponse(
        success=True,
        data=BatchDetailRead(
            **batch_data,
            videos=[VideoRead.model_validate(video) for video in videos],
        ),
        message="Latest batch loaded",
    )


@router.get("/{batch_id}", response_model=ApiResponse[BatchDetailRead])
def get_batch_detail(
    batch_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[BatchDetailRead] | JSONResponse:
    batch = db.get(Batch, batch_id)
    if batch is None:
        return _api_error("Batch not found", status_code=404)

    videos = (
        db.query(Video)
        .filter(Video.batch_id == batch_id)
        .order_by(Video.created_at.asc(), Video.id.asc())
        .all()
    )
    batch_data = BatchRead.model_validate(batch).model_dump()
    return ApiResponse(
        success=True,
        data=BatchDetailRead(
            **batch_data,
            videos=[VideoRead.model_validate(video) for video in videos],
        ),
        message="Batch loaded",
    )
