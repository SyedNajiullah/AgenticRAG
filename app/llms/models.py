from pydantic import BaseModel, Field
from typing import Literal

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