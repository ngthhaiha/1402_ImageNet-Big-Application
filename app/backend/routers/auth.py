from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from backend.database import get_db
from backend.models import User
from backend.schemas import AuthResponse, LoginRequest, RegisterRequest, UserResponse


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(400, "Username already exists")
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(400, "Email already exists")

    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
        created_at=datetime.utcnow().isoformat(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"user_id": user.id, "username": user.username})
    return AuthResponse(
        token=token,
        user=UserResponse(id=user.id, username=user.username, email=user.email),
    )


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = (
        db.query(User)
        .filter(
            (User.username == payload.username_or_email)
            | (User.email == payload.username_or_email)
        )
        .first()
    )
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(401, "Invalid username or password")

    token = create_access_token({"user_id": user.id, "username": user.username})
    return AuthResponse(
        token=token,
        user=UserResponse(id=user.id, username=user.username, email=user.email),
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
    )
