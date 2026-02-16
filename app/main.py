import os
from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.postgres import PostgresSaver
from .schema.pydantic import ChatRequest, ChatResponse
from .graph.edges import create_graph

app = FastAPI(title="Agentic RAG API")

postgres_db_url = os.getenv("POSTGRES_DB_URL")
checkpointer = PostgresSaver.from_conn_string(postgres_db_url)

workflow = create_graph()


@app.get("/")
def root():
    return {"status": "Agentic RAG API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        initial_state = {"question": request.question}
        
        with checkpointer:
            checkpointer.setup()

            chatbot = workflow.compile(checkpointer=checkpointer)
            res = chatbot.invoke(initial_state, config)["messages"][-1].content
            print("Ai res: ", res)

        return ChatResponse(answer=res)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}