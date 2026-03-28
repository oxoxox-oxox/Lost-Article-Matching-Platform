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
    point = Column(Integer, default=0)


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    image = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    user_id = Column(Integer)
    image = Column(String(255))
    description = Column(String(255))  # store the discription of the item
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(Boolean, default=False)
    features_img = Column(Text, nullable=True)  # store embedding as JSON/text
    features_dis = Column(Text, nullable=True)


class Request(Base):
    status = Column(Boolean, default=False)
    features_img = Column(String(255), nullable=True)
    features_dis = Column(String(255), nullable=True)
    __tablename__ = "requests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    image = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(Boolean, default=False)
    user_id = Column(Integer)
    image = Column(String(255))
    dsecription = Column(String(255))  # store the discription of the item
    status = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    features_img = Column(Text, nullable=True)
    features_dis = Column(Text, nullable=True)
    features_img = Column(String(255), nullable=True)
    features_dis = Column(String(255), nullable=True)

# The size of the string might be too small, you can adjust it based on your needs.
