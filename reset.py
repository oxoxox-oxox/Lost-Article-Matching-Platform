import pymysql
from service.database import engine, Base
from service.models.user import User

print("Resetting database...")

# 连接到MySQL服务器
db = pymysql.connect(
    host='localhost',
    user='root',
    password='12345678',
    port=3306
)

cursor = db.cursor()

# 删除旧数据库
try:
    cursor.execute("DROP DATABASE IF EXISTS hackathon_db")
    print("Old database dropped")
except Exception as e:
    print(f"Error dropping database: {str(e)}")

# 创建新数据库
try:
    cursor.execute("CREATE DATABASE hackathon_db")
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
