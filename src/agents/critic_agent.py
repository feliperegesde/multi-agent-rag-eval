from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from src.graph.state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def critic_node(state: AgentState) -> Dict[str, Any]:
    contexts = state.get("retrieved_contexts", [])
    context_str = "\n".join([d["text"] for d in contexts])
    
    prompt = f"""Você é um auditor de fidelidade e alucinação de respostas técnicas.
Contexto recuperado:
{context_str}

Resposta gerada:
{state.get('draft_answer', '')}

A resposta gerada é 100% suportada pelas afirmações do contexto?
Se sim, responda APENAS 'PASS'.
Se a resposta contiver suposições, inventar dados ou alucinar, responda APENAS 'RETRY'."""

    verdict = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    retry_count = state.get("retry_count", 0)
    
    # Máximo de 1 retry para evitar loop infinito
    passed = ("PASS" in verdict) or (retry_count >= 1)
    
    return {
        "critique_verdict": "PASS" if passed else "RETRY",
        "retry_count": retry_count + 1,
        "final_answer": state.get("draft_answer")
    }