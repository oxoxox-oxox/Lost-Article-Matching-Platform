from typing import Optional
from fastapi import APIRouter, Request, Query, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
import service.security as security
import service.models as models

account_router = APIRouter(
    prefix="/account",
    tags=["Account"]
)

templates = Jinja2Templates(directory="templates")


@account_router.get("", response_class=HTMLResponse)
async def account_page(
    request: Request,
    current_user: Optional[models.User] = Depends(security.get_current_user_optional)
):
    if not current_user:
        # Not logged in or invalid token, redirect to login page
        return RedirectResponse(url="/account/login")

    # Logged in, show profile page with current user
    return templates.TemplateResponse(request, "account/profile.html", {"request": request, "user": current_user})


@account_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "account/login.html", {"request": request})


@account_router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(request, "account/register.html", {"request": request})


@account_router.get("/authemail", response_class=HTMLResponse)
async def authemail_page(request: Request, email: str = Query(None)):
    return templates.TemplateResponse(request, "account/authemail.html", {"request": request, "email": email})


@account_router.get("/profile", response_class=HTMLResponse)
async def profile_page(
    request: Request,
    current_user: Optional[models.User] = Depends(security.get_current_user_optional)
):
    # Render profile page with server-side user if present, or clean context for client-side `/auth/me`
    context = {"request": request}
    if current_user:
        context["user"] = current_user
    return templates.TemplateResponse(request, "account/profile.html", context)
