import time
from fastapi import Request, HTTPException

# 内存限流字典：{ "ip_address": [timestamp1, timestamp2, ...] }
IP_RECORDS = {}
RATE_LIMIT = 3       # 限制次数：3 次
TIME_WINDOW = 30     # 限制时间窗口：30 秒内

async def check_rate_limit(request: Request):
    client_ip = request.client.host
    current_time = time.time()

    if client_ip not in IP_RECORDS:
        IP_RECORDS[client_ip] = []

    # 清除时间窗口之外的旧记录
    IP_RECORDS[client_ip] = [t for t in IP_RECORDS[client_ip] if current_time - t < TIME_WINDOW]

    # 判断是否超出频率限制
    if len(IP_RECORDS[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Request rate limit exceeded. Please try again later.")

    # 记录本次请求时间
    IP_RECORDS[client_ip].append(current_time)
    return True

# 蜜罐校验函数
def check_honeypot(honeypot_value: str):
    if honeypot_value:
        # 如果蜜罐字段有值，说明极大概率是无差别填表单的机器人
        raise HTTPException(status_code=400, detail="Bot behavior detected.")