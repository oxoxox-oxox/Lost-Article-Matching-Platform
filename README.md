# Lost Article Matching Platform (LAMP)


## This project is extremely suitable for those who usually lost there items somewhere on campus and don't know where to find them.

### **Lost Article Matching Platform (LAMP) helps students and campus staff recover lost items more efficiently.**

![Project Logo](./logo.jpg)

### Users who find an item can upload item photos and details. Users who lose an item can submit a request with a description and image. The system then matches potential results and helps both sides complete the return process.

## Features

- User registration and login
- Lost/found item reporting
- Item request submission
- Image and information based matching
- Status notifications

## How It Works

1. A user registers and logs in.
2. The finder reports an item with image and description.
3. The owner submits a request for a lost item.
4. The platform searches and recommends matched items.
5. The user confirms the match and arranges pickup.

## Tech Stack

- Backend: FastAPI
- Database: MySQL + SQLAlchemy
- Templates: Jinja2
- Authentication: JWT (`python-jose`, `passlib`)
- AI/Matching related libraries: `openai`, `zhipuai`, `numpy`, `scipy`

## Project Structure

```bash
app/                # Route modules (auth, account, report, search, ai)
service/            # Business logic, DB, security, schemas, models
templates/          # HTML templates
static/             # Static assets (CSS/JS/images)
main.py             # FastAPI application entry point
requirements.txt    # Python dependencies
```

## Prerequisites

- Python 3.9+
- MySQL 8+

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

Create a `.env` file in the project root and configure the following values:

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

OPENAI_API_KEY=<your_openai_or_compatible_api_key>
```

## Run the Project

Start the server with:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- `http://127.0.0.1:8000/`

## Notes

- On startup, the application automatically creates database tables via SQLAlchemy metadata.
- If you want to run on port 80 (as in `main.py`), make sure your environment allows it.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
