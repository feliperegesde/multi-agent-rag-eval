from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from src.graph.state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    prompt = f"""Classifique a seguinte pergunta sobre o FastAPI:
Pergunta: "{state['question']}"

Se a pergunta envolver números de versões, datas de release, histórico de lançamentos ou requisitos de versão do Python, responda "SQL".
Para explicações sobre código, decorators, rotas, dependências, middleware, background tasks ou docker, responda "DOCS".
Responda APENAS a palavra SQL ou DOCS."""
    
    response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    route = "sql" if "SQL" in response else "docs"
    return {"route": route, "retry_count": state.get("retry_count", 0)}