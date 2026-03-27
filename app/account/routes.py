from fastapi import APIRouter, Request, Query, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from service.database import get_db
import service.security as security
import service.models as models

account_router = APIRouter(
    prefix="/account",
    tags=["Account"]
)

templates = Jinja2Templates(directory="templates")

@account_router.get("", response_class=HTMLResponse)
async def account_page(request: Request, db: Session = Depends(get_db)):
    # 从cookie或localStorage获取token
    token = request.cookies.get("token") or request.headers.get("Authorization", "").replace("Bearer ", "")
    
    if not token:
        # 未登录，跳转到登录页
        return RedirectResponse(url="/account/login")
    
    try:
        # 验证token
        payload = security.decode_access_token(token)
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        # 查找用户
        user = db.query(models.User).filter(models.User.email == email).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        # 已登录，显示个人资料页
        return templates.TemplateResponse("account/profile.html", {"request": request, "user": user})
    except:
        # token无效，跳转到登录页
        return RedirectResponse(url="/account/login")

@account_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("account/login.html", {"request": request})

@account_router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("account/register.html", {"request": request})

@account_router.get("/authemail", response_class=HTMLResponse)
async def authemail_page(request: Request, email: str = Query(None)):
    return templates.TemplateResponse("account/authemail.html", {"request": request, "email": email})

@account_router.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    return templates.TemplateResponse("account/profile.html", {"request": request})
