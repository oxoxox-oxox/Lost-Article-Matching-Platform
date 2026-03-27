from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

search_router = APIRouter(
    prefix="/search",
    tags=["Search"]
)
templates = Jinja2Templates(directory="templates")


@search_router.get("/", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse("search/search.html", {"request": request})


@search_router.post("/")
async def search_submit(
    request: Request,
    description: str = Form(...),
    image: UploadFile = File(...)
):
    # 这里可以添加处理上传文件和搜索逻辑
    return templates.TemplateResponse("search/search.html", {"request": request, "success": True})
