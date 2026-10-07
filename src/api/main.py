import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.graph.workflow import app_graph

app = FastAPI(
    title="Multi-Agent RAG with Semantic Routing & Self-Correction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query")
async def execute_query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Pergunta não pode ser vazia.")
    
    result = app_graph.invoke({"question": req.question})
    return {
        "question": req.question,
        "route": result.get("route"),
        "sql_query": result.get("sql_query"),
        "critique_verdict": result.get("critique_verdict"),
        "retries": result.get("retry_count", 0),
        "answer": result.get("final_answer"),
        "sources": [c["source"] for c in result.get("retrieved_contexts", [])]
    }

@app.post("/stream")
async def stream_query(req: QueryRequest):
    async def event_generator():
        # Transmite os eventos de cada nó do LangGraph via SSE
        for event in app_graph.stream({"question": req.question}):
            for node_name, state_update in event.items():
                data = json.dumps({"node": node_name, "data": state_update}, default=str)
                yield f"event: agent_step\ndata: {data}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")