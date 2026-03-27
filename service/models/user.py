from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    username = Column(String(100))
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=False)  # 默认未激活，需要邮箱验证
    verification_code = Column(String(10), nullable=True)
    code_expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow)
    point = Column(Integer, default=0)  # 用户积分，初始为0

    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)  # 举报人ID
    image = Column(String(255))  # 举报的图片URL
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(Boolean, default=False)  # 是否已处理举报
    features = Column(String(255), nullable=True)  # 举报图片的特征值，便于后续分析和处理

    __tablename__ = "requests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)  # 请求人ID
    features = Column(String(255))  # 请求的图片特征值
    created_at = Column(DateTime, default=datetime.utcnow)
