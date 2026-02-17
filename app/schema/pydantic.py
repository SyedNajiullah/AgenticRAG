from pydantic import BaseModel

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

class ThreadHistoryResponse(BaseModel):
    """Schema for returning the full chat history."""
    thread_id: str
    messages: list[Message]
