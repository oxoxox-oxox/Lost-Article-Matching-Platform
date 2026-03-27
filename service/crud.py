#处理数据库的增删改查

from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import service.models as models, service.schemas as schemas, service.security as security

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate, verification_code: str, code_expires_at: datetime):
    hashed_password = security.get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        verification_code=verification_code,
        code_expires_at=code_expires_at
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user_verification(db: Session, user_id: int, is_active: bool = True, verification_code: str = None, code_expires_at: datetime = None):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user:
        db_user.is_active = is_active
        if verification_code:
            db_user.verification_code = verification_code
        if code_expires_at:
            db_user.code_expires_at = code_expires_at
        db.commit()
        db.refresh(db_user)
    return db_user

def verify_code(db: Session, email: str, code: str):
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        return None
    
    # 检查验证码是否正确且在有效期内
    if user.verification_code == code and datetime.utcnow() < user.code_expires_at:
        return user
    return None