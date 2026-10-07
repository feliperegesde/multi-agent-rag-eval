from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from src.graph.state import AgentState
from src.retrieval.hybrid_search import HybridRetriever

retriever = HybridRetriever(docs_folder="data/docs")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def retrieval_node(state: AgentState) -> Dict[str, Any]:
    query_to_search = state["question"]
    docs = retriever.search(query_to_search, top_k=3)
    
    context_str = "\n\n".join([f"Fonte ({d['source']}):\n{d['text']}" for d in docs])
    
    prompt = f"""Responda à pergunta do usuário usando ESTRITAMENTE o contexto abaixo.
Se a informação não estiver presente, declare que a documentação atual não aborda o ponto.

Contexto:
{context_str}

Pergunta: {state['question']}
Resposta técnica e objetiva em português:"""
    
    draft = llm.invoke([HumanMessage(content=prompt)]).content
    return {"retrieved_contexts": docs, "draft_answer": draft}

def rewrite_node(state: AgentState) -> Dict[str, Any]:
    prompt = f"""A consulta '{state['question']}' não retornou contexto suficiente para responder confiavelmente.
Reescreva a pergunta em uma busca mais assertiva com palavras-chave técnicas de FastAPI:"""
    new_q = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {"question": new_q}