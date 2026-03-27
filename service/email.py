import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv
import random
import string

# 指定.env文件的路径，
env_path = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

# 邮件配置
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.163.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")

# 生成随机验证码


def generate_verification_code(length=6):
    """生成指定长度的验证码"""
    characters = string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# 发送验证邮件


def send_verification_email(email, code):
    """发送包含验证码的邮件"""
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL]):
        raise ValueError(
            "SMTP configuration is incomplete. Please check environment variables.")

    # 创建邮件
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "LAMP Account Verification"
    msg['From'] = SENDER_EMAIL
    msg['To'] = email

    # Plain text content
    text = f"""Hello!

Thank you for registering with LAMP (Lost Article Matching Platform).

Your verification code is: {code}

Please use this code within 10 minutes to complete your email verification.

If you did not request this, please ignore this email.

Best regards,
LAMP Team
"""

    # HTML content
    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>LAMP Account Verification</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background-color: #f8fafc;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .email-card {{
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                padding: 40px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .logo {{
                font-size: 24px;
                font-weight: 600;
                color: #3b82f6;
                margin-bottom: 10px;
            }}
            .subtitle {{
                color: #64748b;
                font-size: 16px;
            }}
            .content {{
                margin-bottom: 30px;
            }}
            .greeting {{
                font-size: 18px;
                font-weight: 500;
                color: #1e293b;
                margin-bottom: 20px;
            }}
            .message {{
                color: #334155;
                line-height: 1.6;
                margin-bottom: 20px;
            }}
            .code-container {{
                background-color: #f1f5f9;
                padding: 20px;
                border-radius: 6px;
                text-align: center;
                margin: 30px 0;
            }}
            .code {{
                font-size: 32px;
                font-weight: 700;
                color: #3b82f6;
                letter-spacing: 4px;
            }}
            .note {{
                color: #64748b;
                font-size: 14px;
                margin-top: 10px;
            }}
            .footer {{
                text-align: center;
                color: #94a3b8;
                font-size: 14px;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e2e8f0;
            }}
        </style>
    </head>
    <body>
        <div class=\"container\">
            <div class=\"email-card\">
                <div class=\"header\">
                    <div class=\"logo\">LAMP</div>
                    <div class=\"subtitle\">Lost Article Matching Platform</div>
                </div>
                <div class=\"content\">
                    <div class=\"greeting\">Hello!</div>
                    <div class=\"message\">
                        Thank you for registering with LAMP (Lost Article Matching Platform).<br><br>
                        Please use the following code to complete your email verification:
                    </div>
                    <div class=\"code-container\">
                        <div class=\"code\">{code}</div>
                        <div class=\"note\">This code is valid for 10 minutes.</div>
                    </div>
                    <div class=\"message\">
                        If you did not request this, please ignore this email.
                    </div>
                </div>
                <div class=\"footer\">
                    <p>Best regards,</p>
                    <p>LAMP Team</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    # 添加邮件内容
    part1 = MIMEText(text, 'plain', 'utf-8')
    part2 = MIMEText(html, 'html', 'utf-8')
    msg.attach(part1)
    msg.attach(part2)

    # 发送邮件
    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
        print(f"Success: Email to {email}")
        return True
    except Exception:
        print(f"Failed: Email to {email}")
        return False
