from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from service.database import get_db
from service.models import User
from service.security import decode_access_token


def require_auth_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.headers.get(
        "Authorization", "").replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="No access token provided")

    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid access token")

    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid access token")

    user = db.query(User).filter(User.email == email).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not authenticated")
    return user
