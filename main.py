from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from service.database import engine, Base
from app import auth_router, account_router, ai_router, search_router, report_router
import service.security as security
from sqlalchemy.orm import Session
from service.database import get_db
import service.models as models
import service.schemas as schemas

# 在启动时自动在 MySQL 中创建表结构 (省去手动写 SQL 建表)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Lamp",
    description="Lost Article Matching Platform",
    version="1.0.0"
)

# 配置 CORS 跨域，允许前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 比赛时为方便可设为"*"，生产环境建议指定前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth_router)
app.include_router(account_router)
app.include_router(ai_router)
app.include_router(search_router)
app.include_router(report_router)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置模板渲染
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)