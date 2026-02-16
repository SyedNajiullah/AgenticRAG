from ..vector_db import vector_db
from langchain.tools import tool

@tool("vector_search", description="Search the device manuals and question answer database for relevant documents.")
def vector_search(query: str):
    """Retrieves top 5 results from Qdrant for a given query."""
    try:
        retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={'k': 5})
        result = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in result])
        return context
    except Exception as e:
        return f"Vector search Error: {str(e)}"