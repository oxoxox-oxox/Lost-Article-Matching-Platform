# Lost Article Matching Platform (LAMP)

LAMP is a campus lost-and-found matching platform designed to connect finders and owners faster, improve matching accuracy, and shorten recovery time through automated notifications.

## Highlights

- Account system: registration, email verification code, login, and profile page
- Found item reporting: supports text and image feature storage
- Search assistant: chat-style input with multi-turn history and image upload
- Multimodal matching: text vectors + image vectors with two-stage screening (coarse + refine)
- Auto notification: sends email alerts when high-confidence candidates are found
- Frontend UX: Jinja2-based pages with unified modal/dialog interactions

## Tech Stack

- Backend: FastAPI, Uvicorn
- Database: MySQL, SQLAlchemy
- Templates and static assets: Jinja2, HTML/CSS/JS
- Auth and security: JWT (python-jose), Passlib, rate-limiting and honeypot protection
- AI and vector processing: OpenAI-compatible API, ZhipuAI, NumPy, SciPy, Sentence Transformers, PyTorch, Pillow

## Project Structure

```text
.
|-- app/
|   |-- auth/            # Authentication and verification code routes
|   |-- account/         # Account page routes
|   |-- ai/              # AI chat routes
|   |-- search/          # Search chat, screening, and matching routes
|   |-- report/          # Found-item report and notification routes
|   `-- __init__.py      # Router exports
|-- service/
|   |-- models/          # SQLAlchemy models
|   |-- schemas/         # Pydantic schemas
|   |-- database.py      # DB engine/session setup
|   |-- security.py      # JWT and password utilities
|   |-- email.py         # Email helpers
|   `-- antibot.py       # Anti-abuse strategy
|-- model/               # Additional model and cross-modal engine code
|-- templates/           # Jinja2 templates
|-- static/              # Static assets (CSS/JS/images)
|-- main.py              # Application entry point
|-- reset.py             # MySQL reset script
|-- Dockerfile
|-- docker-compose.yml
`-- requirements.txt
```

## Core Routes

- `/`: Home page
- `/auth/*`: Register, login, verification code, current user
- `/account/*`: Account-related pages (login/register/profile)
- `/report/*`: Found-item report and text/image vector storage
- `/search/*`: Search page, chat search, two-stage screening
- `/ai/chat`: AI chat page and endpoint

## Requirements

- Python 3.10+
- MySQL 8.0+
- Optional: Docker / Docker Compose

## Local Development

1. Clone the repository

```bash
git clone https://github.com/oxoxox-oxox/Lost-Article-Matching-Platform.git
cd Lost-Article-Matching-Platform
```

1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

1. Install dependencies

```bash
pip install -r requirements.txt
```

1. Configure environment variables (create `.env` in the project root)

```env
# Database
DATABASE_URL=mysql+pymysql://<username>:<password>@localhost:3306/<database_name>

# Auth
SECRET_KEY=<your_secret_key>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Email
SMTP_SERVER=<your_smtp_server>
SMTP_PORT=465
SMTP_USERNAME=<your_email_username>
SMTP_PASSWORD=<your_email_password>
SENDER_EMAIL=<your_sender_email>

# AI
OPENAI_API_KEY=<your_openai_compatible_api_key>
ZHIPUAI_API_KEY=<your_zhipuai_api_key>

# Business
MATCH_NOTIFY_THRESHOLD=0.80
```

1. Start the service

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open: <http://127.0.0.1:8000/>

## Run with Docker

1. Prepare `.env` (recommended: configure at least AI keys and SMTP)

1. Build and start

```bash
docker compose up -d --build
```

1. Open: <http://127.0.0.1:8000/>

Useful commands:

```bash
# View app logs
docker compose logs -f app

# Stop services
docker compose down

# Stop and remove DB volume (clear data)
docker compose down -v
```

Default port mapping:

- App: `8000` -> `8000`
- MySQL: `3307` -> `3306`

## Database Reset

If you need to quickly clear and recreate the database schema:

```bash
python reset.py
```

Notes:

- `reset.py` reads `DATABASE_URL`
- MySQL only
- It drops and recreates the target database, so do not run in production unless intended

## Implementation Notes

- On startup, `Base.metadata.create_all(bind=engine)` runs to auto-create tables
- CORS is currently development-friendly (`allow_origins=["*"]`)
- Search chat endpoints return parsed intent + screening results for frontend modal/redirect logic
- Report endpoints can trigger potential-match email notifications when threshold is met

## FAQ

1. Email cannot be sent after startup

- Check whether `SMTP_SERVER/PORT/USERNAME/PASSWORD/SENDER_EMAIL` are fully configured
- Port 465 usually requires SSL; other ports should support STARTTLS

1. AI key errors from APIs

- Confirm both `OPENAI_API_KEY` and `ZHIPUAI_API_KEY` are set in `.env`

1. Database connection failure

- Check `DATABASE_URL`, MySQL service status, account permissions, and port settings

## License

MIT License. See LICENSE.
