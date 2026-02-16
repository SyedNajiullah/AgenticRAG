from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    """Schema for incoming chat requests."""
    question: str
    thread_id: str = "thread-1"

class ChatResponse(BaseModel):
    """Schema for outgoing chat responses."""
    answer: str
