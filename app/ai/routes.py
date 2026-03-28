from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from service.schemas.ai import AIRequest, Message
from openai import OpenAI
import os
import asyncio

# Configure templates
templates = Jinja2Templates(directory="templates")

router = APIRouter(
    prefix="/ai",
    tags=["AI"]
)


async def stream_reasoning_completion(user_question, history, api_key, use_reasoning=False, base_url="https://open.bigmodel.cn/api/paas/v4/"):
    # Construct prompt with reasoning guidance (Core: force AI to output reasoning process)
    prompt_template = """
    
    Please output your reasoning process in detail according to the following structure:
    1. Problem Analysis: Identify core requirements, known conditions, and key points to resolve;
    2. Task Decomposition: Break down the problem step-by-step, explaining the logic, basis, or calculation for each;
    3. Final Result: Provide a clear and concise answer based on the reasoning steps.
    
    Question to answer: {user_question}
    """

    # Decide whether to use reasoning template based on use_reasoning parameter
    if use_reasoning:
        formatted_prompt = prompt_template.format(user_question=user_question)
    else:
        formatted_prompt = user_question

    # Build message history
    messages = []
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": formatted_prompt})

    # Initialize OpenAI compatible client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    try:
        # Initiate streaming request (GLM-4-Flash supports streaming)
        stream = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            stream=True,
            temperature=0.1,
            max_tokens=8192
        )

        # Receive chunks and generate reasoning process (Core of streaming output)
        for chunk in stream:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                yield chunk_content
                await asyncio.sleep(0.01)  # Control flow speed
    except Exception as e:
        error_message = f"Call failed: {str(e)}"
        yield error_message
        # Common error prompts: API Key error, network issues, concurrency limit (GLM-4-Flash limited to 30)
        if "invalid_api_key" in str(e).lower():
            yield "\nAPI Key Error"
        elif "rate_limit" in str(e).lower():
            yield "\nConcurrency limit reached, please try again later!"


@router.post("/chat")
async def chat(request: AIRequest):
    # Get API Key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured")

    # Return streaming response
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
    return templates.TemplateResponse(request, "ai_chat.html", {"request": request})
