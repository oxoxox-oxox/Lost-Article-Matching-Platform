from fastapi import APIRouter, Depends, HTTPException, Request
from service.email import send_verification_email, generate_verification_code
from service.antibot import check_rate_limit, check_honeypot
import service.database as database
from service.schemas.user import UserCreate, UserLogin, UserResponse, Token, VerificationCodeRequest, VerifyCodeRequest
import service.crud as crud
from datetime import datetime, timedelta
from fastapi import Request, Depends, HTTPException
from service.database import get_db
import service.models as models
import service.schemas as schemas
import service.security as security
from sqlalchemy.orm import Session


auth_router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@auth_router.get("/me", response_model=schemas.UserResponse)
def get_current_user(
    request: Request,
    db: Session = Depends(get_db)
):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        raise HTTPException(status_code=401, detail="No access token provided")
    try:
        payload = security.decode_access_token(token)
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid access token")
    except:
        raise HTTPException(status_code=401, detail="Invalid access token")
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@auth_router.post("/register", response_model=UserResponse)
def register(
    user: UserCreate,
    request: Request,
    db: Session = Depends(database.get_db),
    rate_limit: bool = Depends(check_rate_limit)  # 注入限流防刷
):
    print(f"User information: email={user.email}, username={user.username}")

    try:
        # 1. 检查蜜罐
        check_honeypot(user.honeypot)

        # 2. 检查邮箱是否已注册
        db_user = crud.get_user_by_email(db, email=user.email)
        if db_user and db_user.is_active:
            raise HTTPException(
                status_code=400, detail="Email already registered.")

        # 3. 生成验证码
        verification_code = generate_verification_code()
        code_expires_at = datetime.utcnow() + timedelta(minutes=10)
        print(
            f"Verification code generated: {verification_code}, expires_at: {code_expires_at}")

        # 4. 创建用户（未激活状态）或更新已有未激活用户
        if db_user and not db_user.is_active:
            db_user.username = user.username
            db_user.hashed_password = security.get_password_hash(user.password)
            db_user.verification_code = verification_code
            db_user.code_expires_at = code_expires_at
            db_user.is_active = False
            db.commit()
            db.refresh(db_user)
            print(f"Unverified user updated, ID: {db_user.id}")
        else:
            db_user = crud.create_user(
                db=db, user=user, verification_code=verification_code, code_expires_at=code_expires_at)
            print(f"User created successfully, ID: {db_user.id}")

        # 5. 发送验证邮件
        try:
            print(
                f"调用send_verification_email({user.email}, {verification_code})")
            email_sent = send_verification_email(user.email, verification_code)
            if not email_sent:
                # 邮件发送失败，回滚本次注册流程
                if not db_user.is_active and db_user.verification_code == verification_code:
                    db_user.verification_code = None
                    db_user.code_expires_at = None
                    db.commit()
                raise HTTPException(
                    status_code=500, detail="Email verification failed, please try again later")
        except Exception as email_error:
            # 邮件发送异常，回滚本次验证码状态
            import traceback
            traceback.print_exc()
            if not db_user.is_active and db_user.verification_code == verification_code:
                db_user.verification_code = None
                db_user.code_expires_at = None
                db.commit()
            raise HTTPException(
                status_code=500, detail="Email verification failed, please try again later")
        return db_user
    except HTTPException:
        # 重新抛出HTTPException，保持原有行为
        print("HTTPException")
        raise
    except Exception as e:
        # 记录其他异常
        print(f"Registration failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        # 回滚数据库事务
        db.rollback()
        # 抛出通用错误
        raise HTTPException(
            status_code=500, detail="Registration failed, please try again later")


@auth_router.post("/resend-code")
def resend_verification_code(
    request: VerificationCodeRequest,
    db: Session = Depends(database.get_db),
    rate_limit: bool = Depends(check_rate_limit)
):
    # 1. 查找用户
    user = crud.get_user_by_email(db, email=request.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 2. 检查用户是否已激活
    if user.is_active:
        raise HTTPException(status_code=400, detail="Account already verified")

    # 3. 生成新的验证码
    verification_code = generate_verification_code()
    code_expires_at = datetime.utcnow() + timedelta(minutes=10)

    # 4. 更新用户验证码
    crud.update_user_verification(
        db, user.id, verification_code=verification_code, code_expires_at=code_expires_at)

    # 5. 发送验证邮件
    email_sent = send_verification_email(request.email, verification_code)
    if not email_sent:
        raise HTTPException(
            status_code=500, detail="Email verification failed, please try again later")

    return {"message": "Verification code resent"}


@auth_router.post("/verify-code")
def verify_code(
    request: VerifyCodeRequest,
    db: Session = Depends(database.get_db)
):
    # 1. 验证验证码
    user = crud.verify_code(db, email=request.email, code=request.code)
    if not user:
        raise HTTPException(
            status_code=400, detail="Verification code error or expired")

    # 2. 激活用户
    crud.update_user_verification(
        db, user.id, is_active=True, verification_code=None, code_expires_at=None)

    return {"message": "Account verified"}


@auth_router.post("/login", response_model=Token)
def login(
    user_credentials: UserLogin,
    request: Request,
    db: Session = Depends(database.get_db),
    rate_limit: bool = Depends(check_rate_limit)  # 登录同样防暴力破解
):
    # 1. 仅允许邮箱登录
    login_email = (user_credentials.username_or_email or "").strip()
    if '@' not in login_email:
        raise HTTPException(
            status_code=401, detail="Email login only")

    user = crud.get_user_by_email(db, email=login_email)

    if not user:
        raise HTTPException(
            status_code=401, detail="Email not found")

    # 2. 检查用户是否已激活
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account not verified")

    # 3. 校验密码
    if not security.verify_password(user_credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Password error")

    # 4. 签发 Token（始终使用邮箱作为sub）
    access_token = security.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@auth_router.get("/check-username")
def check_username(
    username: str,
    db: Session = Depends(database.get_db)
):
    user = crud.get_user_by_username(db, username=username)
    return {"exists": user and user.is_active}  # 只有已激活的账户才算占用


@auth_router.get("/check-email")
def check_email(
    email: str,
    db: Session = Depends(database.get_db)
):
    user = crud.get_user_by_email(db, email=email)
    # Only activated accounts are treated as occupied for registration checks.
    return {"exists": bool(user and user.is_active)}
