import sqlite3
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from src.graph.state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def sql_node(state: AgentState) -> Dict[str, Any]:
    schema_info = "Tabela: releases (colunas: version TEXT, release_date TEXT, python_min_version TEXT, key_features TEXT)"
    prompt = f"""Você é um especialista em SQLite. Gere uma consulta SELECT para responder à pergunta:
Schema: {schema_info}
Pergunta: {state['question']}

Retorne EXCLUSIVAMENTE a consulta SQL pura, sem blocos markdown (```sql) ou explicações."""
    
    query = (
        llm.invoke([HumanMessage(content=prompt)])
        .content.strip()
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )
    
    # Guardrail de segurança: apenas SELECT
    if not query.lower().startswith("select"):
        return {
            "sql_query": query,
            "sql_result": "Erro de segurança: Apenas consultas SELECT são permitidas.",
            "final_answer": "Operação bloqueada por motivos de segurança."
        }
        
    try:
        conn = sqlite3.connect("data/fastapi_metadata.db")
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        result_str = str(rows)
    except Exception as e:
        result_str = f"Erro na execução da consulta: {e}"
        
    synth_prompt = f"""Pergunta original: {state['question']}
Query executada: {query}
Dados retornados: {result_str}

Responda de forma clara, técnica e objetiva em português."""
    draft = llm.invoke([HumanMessage(content=synth_prompt)]).content
    
    return {
        "sql_query": query,
        "sql_result": result_str,
        "draft_answer": draft,
        "critique_verdict": "PASS",
        "final_answer": draft
    }