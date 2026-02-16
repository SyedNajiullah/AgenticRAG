from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from mem0 import MemoryClient
from dotenv import load_dotenv

load_dotenv("../.env")

llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
mem0 = MemoryClient()