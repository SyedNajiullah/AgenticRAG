from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GreetingClassification(BaseModel):
    category: Literal["greeting", "non_greeting"] = Field(
        description="Whether the message is a greeting or non-greeting"
    )

class MemoryClassifcation(BaseModel):
    category: Literal["self_memory", "general_knowledge"] = Field(
        description="Whether the message is a memory base question or knowledge base question"
    )

class AnswerSatisfactionClassification(BaseModel):
    category: Literal["yes", "no"] = Field(
        description="Whether the message is yes or no"
    )

class ChatState(TypedDict):
    question: str # question from the user
    messages: Annotated[List[BaseMessage], add_messages] # all messages history fot short term memory.
    phase: str # current phase of the chat web or rag
    rag_context: str # context from the rag
    web_context: str # context from the web
    ltm_context: str # context from long term memory