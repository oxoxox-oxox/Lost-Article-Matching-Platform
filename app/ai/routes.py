from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from service.schemas.ai import AIRequest, Message
from openai import OpenAI
import os
import asyncio

# 配置模板
templates = Jinja2Templates(directory="templates")

router = APIRouter(
    prefix="/ai",
    tags=["AI"]
)

async def stream_reasoning_completion(user_question, history, api_key, use_reasoning=False, base_url="https://open.bigmodel.cn/api/paas/v4/"):
    # 构造带推理引导的prompt（核心：强制AI输出推理过程）
    prompt_template = """
    
    请按照以下结构详细输出你的推理过程：
    1. 问题分析：明确问题的核心需求、已知条件、需要解决的关键点；
    2. 任务拆解：分步骤拆解问题，说明每一步的思考逻辑、依据或计算过程；
    3. 最终结果：基于推理步骤给出明确、简洁的答案。
    
    需要解答的问题：{user_question}
    """

    # 根据use_reasoning参数决定是否使用推理模板
    if use_reasoning:
        formatted_prompt = prompt_template.format(user_question=user_question)
    else:
        formatted_prompt = user_question
    
    # 构建消息历史
    messages = []
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": formatted_prompt})
    
    # 初始化OpenAI兼容客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    try:
        # 发起流式请求（GLM-4-Flash支持流式）
        stream = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            stream=True,
            temperature=0.1,
            max_tokens=8192
        )
        
        # 逐块接收并生成推理过程（流式输出核心）
        for chunk in stream:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                yield chunk_content
                await asyncio.sleep(0.01)  # 控制流速度
    except Exception as e:
        error_message = f"调用失败：{str(e)}"
        yield error_message
        # 常见错误提示：API Key错误、网络问题、并发超限（GLM-4-Flash限30并发）
        if "invalid_api_key" in str(e).lower():
            yield "\nAPI Key Error"
        elif "rate_limit" in str(e).lower():
            yield "\n并发请求超限，请稍后重试！"

@router.post("/chat")
async def chat(request: AIRequest):
    # 从环境变量获取API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured")
    
    # 返回流式响应
    return StreamingResponse(
        stream_reasoning_completion(
            user_question=request.user_question,
            history=request.history,
            api_key=api_key,
            use_reasoning=request.use_reasoning
        ),
        media_type="text/plain"
    )

@router.get("/chat", response_class=HTMLResponse)
async def ai_chat_page(request: Request):
    return templates.TemplateResponse("ai_chat.html", {"request": request})