import os
from .utils import get_rag_docs, COLLECTION_NAME
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_DB_URL"), 
    api_key=os.getenv("QDRANT_API_KEY")
)

collections = [c.name for c in qdrant_client.get_collections().collections]
vector_db = None


if COLLECTION_NAME not in collections: # collection doesnot exists
    print("Collection does not exists: creating collection -> inserting vectors")

    docs = get_rag_docs()

    vector_db = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=os.getenv("QDRANT_DB_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=True,
        collection_name="project",
    )
else: # collection exists
    points = qdrant_client.count("project").count

    if points == 0: # collection exists but vectors does not exist in the collection.
        print("Collection is empty -> inserting vectors in colelction")

        docs = get_rag_docs()
        
        vector_db = QdrantVectorStore.from_documents(
            docs,
            embeddings,
            url=os.getenv("QDRANT_DB_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=COLLECTION_NAME
        )
    else: # collection and vectors both exists.
        print("Both collection and vectors exists")
        vector_db = QdrantVectorStore(
            client=qdrant_client,
            collection_name="project",  # Your collection name
            embedding=embeddings
        )