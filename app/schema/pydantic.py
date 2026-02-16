from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    """Schema for incoming chat requests."""
    question: str
    thread_id: str = "thread-1"

class ChatResponse(BaseModel):
    """Schema for outgoing chat responses."""
    answer: str

class ThreadRequest(BaseModel):
    """Schema for retrieving all chat on a specific thread."""
    thread_id: str

class Message(BaseModel):
    """Standard message format for the API."""
    role: str # 'user', 'assistant', or 'system'
    content: str

class ThreadHistoryResponse(BaseModel):
    """Schema for returning the full chat history."""
    thread_id: str
    messages: list[Message]
