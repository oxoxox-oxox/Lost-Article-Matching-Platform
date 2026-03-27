from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

report_router = APIRouter(
    prefix="/report",
    tags=["Report"]
)
templates = Jinja2Templates(directory="templates")


@report_router.get("/", response_class=HTMLResponse)
async def report_page(request: Request):
    return templates.TemplateResponse("report/report.html", {"request": request})


@report_router.post("/")
async def report_submit(
    request: Request,
    image: UploadFile = File(...),
    time: str = Form(...),
    location: str = Form(...),
    description: str = Form(...)
):
    # 这里可以添加处理上传文件和存储信息的逻辑
    return templates.TemplateResponse("report/report.html", {"request": request, "success": True})
