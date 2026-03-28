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

# Automatically create table structure in MySQL on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Lamp",
    description="Lost Article Matching Platform",
    version="1.0.0"
)

# Configure CORS cross-origin, allowing front-end calls
app.add_middleware(
    CORSMiddleware,
    # Set to "*" for convenience during competition; production suggested to specify front-end domain
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(auth_router)
app.include_router(account_router)
app.include_router(ai_router)
app.include_router(search_router)
app.include_router(report_router)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure template rendering
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
