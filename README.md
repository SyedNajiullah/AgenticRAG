# AgenticRAG Assistant 🚀

![Agentic RAG Architecture](diagram/Agentic_RAG.png)

An  **Agentic Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **LangGraph**, **LangChain**, **Qdrant**, **Mem0**, **Tavily**, and **PostgreSQL**. This agent reasons about the best information source—dynamically switching between **Long-Term Memory**, **Vector Knowledge Bases**, and **Real-time Web Search**.

---

## 🧠 System Architecture

The agent follows a stateful reasoning flow powered by **LangGraph**:

```mermaid
graph TD
    A[User Request] --> B{Planning Node}
    B -->|Greeting / Personal| C[Fall-back LLM]
    B -->|General Inquiry| D[Context Search - Qdrant]
    D --> E{Relevancy Check}
    E -->|Success| F[Generate Answer & Save to Mem0]
    E -->|Insufficient Info| G[Web Search - Tavily]
    G --> H{Final Relevancy Check}
    H -->|Success| F
    H -->|Fail| C
    C --> F
    F --> I[Response to User]
```

### Key Technologies:
- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
- **API Framework**: FastAPI
- **LLM**: Groq (Llama 3)
- **Vector Database**: Qdrant
- **Personalized Memory**: Mem0 (Long-Term) & PostgreSQL (Short-Term/Checkpoints)
- **Search**: Tavily AI (Web search)

---

## 📂 Project Structure

```text
AgenticRAG/
├── app/
│   ├── graph/           # LangGraph reasoning logic (nodes & edges)
│   │   ├── nodes.py     # Agent logic for each node (Planning, Search, etc.)
│   │   └── edges.py     # Logic for routing between nodes
│   ├── llms/            # LLM engine configurations and State models
│   │   ├── engine.py    # LLM Initialization (Groq, Structured Output)
│   │   └── models.py    # Graph State definitions
│   ├── schema/          # Pydantic models for API validation
│   ├── tools/           # Retrieval tools
│   │   ├── rag.py       # Qdrant vector search tool
│   │   └── web.py       # Tavily web search tool
│   ├── datasets/        # Source documents for the knowledge base
│   ├── main.py          # FastAPI entry point & Endpoint definitions
│   ├── utils.py         # Helper functions (Memory IDs, etc.)
│   └── vector_db.py     # Qdrant client & Collection management
├── diagram/             # Architecture diagrams and assets
├── notebooks/           # Project and Preprocessing notebooks
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🛠️ Setup & Installation

### 1. Configure Environment
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
QDRANT_API_KEY=your_qdrant_key
QDRANT_DB_URL=your_qdrant_url
MEM0_API_KEY=your_mem0_key
POSTGRES_DB_URL=postgresql://user:password@localhost:5432/dbname
```

### 2. Install Dependencies
```bash
python -m venv my-venv
source my-venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running the API
```bash
uvicorn app.main:app --reload
```
The server will start at `http://127.0.0.1:8000/`.

### Core API Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/chat` | Submit a question to the agentic flow. |
| `POST` | `/history` | Fetch chat history for a specific `thread_id`. |
| `GET` | `/docs` | Interactive Swagger API documentation. |

---