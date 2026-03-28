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
    # Logic to handle file uploads and search can be added here
    return templates.TemplateResponse("search/search.html", {"request": request, "success": True})
