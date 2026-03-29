# Lost Article Matching Platform (LAMP)

LAMP is a campus lost-and-found platform that helps connect item finders and owners quickly. It combines text and image understanding to improve match quality and sends automatic notifications when high-confidence matches are detected.

![image](logo.jpg)

## Key Features

- User accounts: registration, verification code flow, login, and profile page
- Reporting flow: submit found-item information with text and image features
- Search assistant: chat-style search with conversation history and image upload
- Multimodal matching: text and image vectors with two-stage filtering (coarse and refine)
- Email alerts: automatic notifications for potential high-confidence matches
- Web UI: Jinja2 templates with consistent modal and dialog interactions

## How Fine Screening Works

We use a two-stage matching pipeline to keep retrieval efficient while improving precision.

### 1. Coarse screening

- We first compute **vector similarity** between the user query and candidate items.
- This stage quickly retrieves a candidate pool and **removes low-similarity noise**.

### 2. Fine screening (refinement)

- We run fine screening only on top candidates from stage one.
- Depending on the endpoint, refinement is done in one of two ways:
  - LLM-based semantic judgment: the model labels each pair as yes, maybe, or no.
  - Multimodal fine ranking: we combine text-to-text, image-to-text (with calibrated scaling), and image-to-image similarity.

Final score fusion

- In LLM refinement, the final score is a **weighted fusion**:
  - final_score = 0.7 *coarse_score + 0.3* llm_score
- In multimodal refinement, the final score is the arithmetic mean of the collected modality scores.

Output and ranking

- Every refined candidate receives a **final_score** and a **refine_label**.
- Results are sorted by final_score in descending order, and only the top matches are returned.

## Technology Stack

- `Backend`: FastAPI, Uvicorn
- `Database`: MySQL, SQLAlchemy
- `Frontend` rendering: Jinja2 templates, HTML, CSS, JavaScript
- `Authentication and security`: JWT (python-jose), Passlib, rate-limiting, honeypot checks
- `AI and vector processing`: OpenAI-compatible API, ZhipuAI, NumPy, SciPy, Sentence Transformers, PyTorch, Pillow

## Project Layout

```text
.
|-- app/
|   |-- auth/            # Authentication and verification routes
|   |-- account/         # Account page routes
|   |-- ai/              # AI chat routes
|   |-- search/          # Search flow, filtering, and matching routes
|   |-- report/          # Found-item reporting and notification routes
|   `-- __init__.py      # Router exports
|-- service/
|   |-- models/          # SQLAlchemy models
|   |-- schemas/         # Pydantic schemas
|   |-- database.py      # Database engine and session setup
|   |-- security.py      # JWT and password helpers
|   |-- email.py         # Email utilities
|   `-- antibot.py       # Anti-abuse logic
|-- model/               # Extra model and cross-modal engine code
|-- templates/           # Jinja2 templates
|-- static/              # Static assets (CSS, JS, images)
|-- main.py              # Application entry point
|-- reset.py             # Database reset script for MySQL
|-- Dockerfile
|-- docker-compose.yml
`-- requirements.txt
```

## Main Routes

- `/`: Home page
- `/auth/*`: Register, login, verification, current-user endpoints
- `/account/*`: Account-related pages (login, register, profile)
- `/report/*`: Found-item reporting and vector persistence
- `/search/*`: Search page, chat search, and two-stage filtering
- `/ai/chat`: AI chat page and API endpoint

## Requirements

- Python 3.10 or newer
- MySQL 8.0 or newer
- Optional: Docker and Docker Compose

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/oxoxox-oxox/Lost-Article-Matching-Platform.git
cd Lost-Article-Matching-Platform
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file in the project root

```env
# Database
DATABASE_URL=mysql+pymysql://<username>:<password>@localhost:3306/<database_name>

# Authentication
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

### 5. Start the application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open <http://127.0.0.1:8000/>.

## Docker Deployment

### 1. Prepare the `.env` file (at minimum, configure AI keys and SMTP values)

### 2. Build and start containers

```bash
docker compose up -d --build
```

### 3. Open <http://127.0.0.1:8000/>

Useful commands:

```bash
# Follow app logs
docker compose logs -f app

# Stop services
docker compose down

# Stop services and remove DB volume
docker compose down -v
```

Default port mapping:

- App: `8000` -> `8000`
- MySQL: `3307` -> `3306`

## Database Reset

To drop and recreate the target database schema quickly:

```bash
python reset.py
```

Notes:

- `reset.py` reads `DATABASE_URL` from `.env`
- MySQL only
- Do not run in production unless you intentionally want to wipe database data

## Runtime Notes

- The app initializes tables at startup via `Base.metadata.create_all(bind=engine)`
- CORS currently allows all origins for development convenience
- Search chat endpoints return parsed intent and filtering results for frontend flow control
- Report endpoints can trigger notification emails when similarity exceeds the configured threshold

## Troubleshooting

### 1. Email sending fails at startup

- Verify `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, and `SENDER_EMAIL`
- Port 465 typically requires SSL; other ports may require STARTTLS

### 2. AI key authentication errors

- Verify both `OPENAI_API_KEY` and `ZHIPUAI_API_KEY` in `.env`

### 3. Database connection errors

- Verify `DATABASE_URL`, MySQL status, user permissions, and host/port values

## License

This project is licensed under the MIT License. See the LICENSE file for details.
