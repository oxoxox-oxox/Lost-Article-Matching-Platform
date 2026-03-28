from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    username = Column(String(100))
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=False)
    verification_code = Column(String(10), nullable=True)
    code_expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow)
    point = Column(Integer, default=0)


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    image = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(Boolean, default=False)
    features_img = Column(Text, nullable=True)  # store embedding as JSON/text
    features_dis = Column(Text, nullable=True)


class Request(Base):
    __tablename__ = "requests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    image = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Embedding vectors are long JSON arrays, so Text is required.
    features_img = Column(Text, nullable=True)
    features_dis = Column(Text, nullable=True)
