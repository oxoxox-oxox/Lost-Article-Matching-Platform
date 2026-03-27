from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

# 注册时的请求体，包含蜜罐字段
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    # 蜜罐字段：普通用户前端隐藏不填，爬虫通常会乱填(狗头)
    honeypot: Optional[str] = None

# 登录时的请求体
class UserLogin(BaseModel):
    username_or_email: str
    password: str

# 验证码请求体
class VerificationCodeRequest(BaseModel):
    email: EmailStr

# 验证码验证请求体
class VerifyCodeRequest(BaseModel):
    email: EmailStr
    code: str

# 返回给前端的用户信息（不包含密码）
class UserResponse(BaseModel):
    id: int
    email: EmailStr
    username: str
    is_active: bool

    # 配置 Pydantic 模型从 SQLAlchemy 模型中读取数据 (启用 ORM 模式)
    class Config:
        from_attributes = True

# Token 响应模型
class Token(BaseModel):
    access_token: str
    token_type: str