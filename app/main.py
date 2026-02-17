import os
from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.postgres import PostgresSaver
from .schema.pydantic import ChatRequest, ChatResponse, ThreadHistoryResponse, ThreadRequest, Message
from .graph.edges import create_graph

app = FastAPI(title="Agentic RAG API")

workflow = create_graph()


@app.get("/")
def root():
    return {"status": "Agentic RAG API is running"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        initial_state = {"question": request.question}
        
        with PostgresSaver.from_conn_string(os.getenv("POSTGRES_DB_URL")) as cp:
            cp.setup()

            chatbot = workflow.compile(checkpointer=cp)
            res = chatbot.invoke(initial_state, config)["messages"][-1].content
            print("Ai res: ", res)

        return ChatResponse(answer=res)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/history", response_model=ThreadHistoryResponse)
def get_chat_history(request: ThreadRequest):

    try:

        config = {"configurable": {"thread_id": request.thread_id}}

        with PostgresSaver.from_conn_string(os.getenv("POSTGRES_DB_URL")) as cp:

            g = workflow.compile(checkpointer=cp)
        
            snap = g.get_state(config)
            msgs = snap.values.get("messages", [])

            formatted_messages = []
            for msg in msgs:
                # Map LangChain message types to roles
                role = "user" if msg.type == "human" else "assistant" if msg.type == "ai" else msg.type
                
                if role in ["user", "assistant"]:
                    formatted_messages.append(Message(role=role, content=msg.content))
                    
            return ThreadHistoryResponse(
                thread_id=request.thread_id,
                messages=formatted_messages
            )
    except Exception as e:
        print(f"History Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))