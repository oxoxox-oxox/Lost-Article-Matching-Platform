import os
import pymysql
from dotenv import load_dotenv
from sqlalchemy.engine.url import make_url
from service.database import engine, Base

print("Resetting database...")

load_dotenv()
database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise RuntimeError("DATABASE_URL is not set in .env")

parsed = make_url(database_url)
db_name = parsed.database

if parsed.get_backend_name() != "mysql":
    raise RuntimeError("reset.py only supports MySQL DATABASE_URL")

if not parsed.username:
    raise RuntimeError("DATABASE_URL must include a MySQL username")

if not db_name:
    raise RuntimeError("DATABASE_URL must include a database name")

# 连接到MySQL服务器
db = pymysql.connect(
    host=parsed.host or "localhost",
    user=parsed.username,
    password=parsed.password,
    port=parsed.port or 3306,
    charset="utf8mb4",
    autocommit=True,
)

cursor = db.cursor()

# 删除旧数据库
try:
    cursor.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
    print("Old database dropped")
except Exception as e:
    print(f"Error dropping database: {str(e)}")

# 创建新数据库
try:
    cursor.execute(f"CREATE DATABASE `{db_name}`")
    print("New database created")
except Exception as e:
    print(f"Error creating database: {str(e)}")

# 关闭连接
db.close()

# 创建表结构
print("Creating tables...")
try:
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully")
except Exception as e:
    print(f"Error creating tables: {str(e)}")

print("Database reset completed!")
