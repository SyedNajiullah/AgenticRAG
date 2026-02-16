import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .llms.models import ChatState

def get_rag_docs():
    # loading
    all_docs = []
    for csv_path in ["../datasets/data_preprocessed/device_manuals.csv", "../datasets/data_preprocessed//question_answer.csv"]:
        if os.path.exists(csv_path):
            loader = CSVLoader(file_path=csv_path, encoding="latin1")
            docs = loader.load()  # Each row = 1 Document
            all_docs.extend(docs)  # Keep Document objects
    # chuncking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(all_docs)
    
    return texts

def return_phase(state: ChatState):
    """Routing function for check_relevence conditional edges"""
    return state["phase"]

COLLECTION_NAME = "project"
memo_id = "naji"