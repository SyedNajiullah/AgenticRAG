from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from mem0 import MemoryClient
from dotenv import load_dotenv
from .models import GreetingClassification, MemoryClassifcation, AnswerSatisfactionClassification
import os

# Load .env from project root
load_dotenv() 

# Debugging: Print first 5 chars of the key to verify it's loaded correctly
key = os.getenv("GROQ_API_KEY")
if key:
    print(f"DEBUG: GROQ_API_KEY loaded, starts with: {key[:5]}...")
else:
    print("DEBUG: GROQ_API_KEY NOT FOUND in environment!")

mem_key = os.getenv("MEM0_API_KEY")
if mem_key:
    print(f"DEBUG: MEM0_API_KEY loaded, starts with: {mem_key[:5]}...")
else:
    print("DEBUG: MEM0_API_KEY NOT FOUND in environment!")

tav_key = os.getenv("TAVILY_API_KEY")
if tav_key:
    print(f"DEBUG: TAVILY_API_KEY loaded, starts with: {tav_key[:5]}...")
else:
    print("DEBUG: TAVILY_API_KEY NOT FOUND in environment!")

# Use a valid Groq model name
llm = ChatGroq(model="llama-3.3-70b-versatile")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
mem0 = MemoryClient()

structured_llm = llm.with_structured_output(GreetingClassification, method="json_mode")
structured_llm_answer = llm.with_structured_output(AnswerSatisfactionClassification, method="json_mode")
structured_llm_memory = llm.with_structured_output(MemoryClassifcation, method="json_mode")