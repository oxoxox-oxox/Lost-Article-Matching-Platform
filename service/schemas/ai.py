from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class AIRequest(BaseModel):
    user_question: str
    history: Optional[List[Message]] = []
    use_reasoning: bool = False

class AIResponse(BaseModel):
    content: str