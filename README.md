# Lost Article Matching Platform (LAMP)

Lost Article Matching Platform helps students and campus staff recover lost items more efficiently.

Users who find an item can upload photos and details. Users who lose an item can submit a request with text and image information. The system uses text and image embeddings to find potential matches and notify users when confidence is high.

## Features

- User registration, email verification, and login
- Account pages for login, registration, and profile display
- Lost item reporting workflow
- Lost item search workflow
- Text and image embedding for matching
- Two-stage screening for candidate retrieval
- AI chat endpoint for Q&A/reasoning output
- Email notifications for verification and high-confidence matches

## Tech Stack

- Backend: FastAPI, Uvicorn
- Database: MySQL, SQLAlchemy
- Templates: Jinja2
- Auth and security: JWT (python-jose), password hashing (passlib), anti-bot rate limiting
- AI and matching: OpenAI-compatible client, ZhipuAI, NumPy, SciPy, Sentence Transformers, PyTorch, Pillow

## Project Structure

```text
|-- app/
|   |-- account/          # Account page routes
|   |-- ai/               # AI chat routes
|   |-- auth/             # Auth and verification routes
|   |-- report/           # Report item routes and matching notification logic
|   |-- search/           # Search routes and two-stage screening
|   `-- __init__.py       # Router exports
|-- model/
|   |-- Cross-Modal_Finder/
|   `-- item_embedding_engine/
|-- service/
|   |-- models/           # SQLAlchemy models
|   |-- schemas/          # Pydantic schemas
|   |-- database.py       # DB engine/session setup
|   |-- crud.py           # Data access logic
|   |-- security.py       # JWT and password utilities
|   |-- email.py          # Email sending helpers
|   `-- antibot.py        # Basic anti-abuse checks
|-- static/               # CSS/JS/assets
|-- templates/            # Jinja2 HTML templates
|-- main.py               # FastAPI app entry point
|-- reset.py              # Optional reset script
`-- requirements.txt      # Python dependencies
```

## Main Routes

- Home page: /
- Authentication: /auth
- Account pages: /account
- Report pages and APIs: /report
- Search pages and APIs: /search
- AI chat page and API: /ai/chat

## Prerequisites

- Python 3.9 or later
- MySQL 8.0 or later
- Docker Desktop 

## Installation

1. Clone the repository:

```bash
git clone https://github.com/oxoxox-oxox/Lost-Article-Matching-Platform.git
cd Lost-Article-Matching-Platform
```

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a .env file in the project root:

```env
DATABASE_URL=mysql+pymysql://<username>:<password>@localhost:3306/<database_name>

SECRET_KEY=<your_secret_key>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

SMTP_SERVER=<your_smtp_server>
SMTP_PORT=465
SMTP_USERNAME=<your_email_username>
SMTP_PASSWORD=<your_email_password>
SENDER_EMAIL=<your_sender_email>

OPENAI_API_KEY=<your_openai_compatible_api_key>
ZHIPUAI_API_KEY=<your_zhipuai_api_key>

MATCH_NOTIFY_THRESHOLD=0.80
```

Notes:

- OPENAI_API_KEY is used by the AI chat module.
- ZHIPUAI_API_KEY is used for multimodal summary and embedding generation in report/search workflows.
- MATCH_NOTIFY_THRESHOLD controls when automatic match notification emails are sent.
- For Docker Compose, `DATABASE_URL` is injected automatically and points to the `db` service.

## Run the Project

Recommended (development):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- <http://127.0.0.1:8000/>

Alternative:

- Running python main.py starts Uvicorn on port 80 as defined in main.py.

## Run with Docker

1. (Optional) Keep your existing `.env` and set business/API keys there, such as `OPENAI_API_KEY`, `ZHIPUAI_API_KEY`, and email settings.

1. Build and start services:

```bash
docker compose up -d --build
```

1. Open the app:

- <http://127.0.0.1:8000/>

Useful commands:

```bash
# View logs
docker compose logs -f app

# Stop all services
docker compose down

# Stop and remove DB volume (clears data)
docker compose down -v
```

Default ports:

- App: `8000` (host) -> `8000` (container)
- MySQL: `3307` (host) -> `3306` (container)

## Development Notes

- On startup, SQLAlchemy creates tables automatically using Base.metadata.create_all.
- CORS is currently configured to allow all origins for easier development.
- Route modules are registered in main.py via router exports from app/__init__.py.

## License

This project is licensed under the MIT License. See LICENSE for details.
