from typing import Literal
from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.agents.supervisor import supervisor_node
from src.agents.sql_agent import sql_node
from src.agents.retrieval_agent import retrieval_node, rewrite_node
from src.agents.critic_agent import critic_node

def route_decision(state: AgentState) -> Literal["sql_node", "retrieval_node"]:
    return "sql_node" if state.get("route") == "sql" else "retrieval_node"

def critique_decision(state: AgentState) -> Literal["rewrite_node", "__end__"]:
    if state.get("critique_verdict") == "PASS":
        return END
    return "rewrite_node"

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("sql_node", sql_node)
    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("critic_node", critic_node)
    workflow.add_node("rewrite_node", rewrite_node)
    
    workflow.set_entry_point("supervisor_node")
    workflow.add_conditional_edges("supervisor_node", route_decision)
    
    workflow.add_edge("sql_node", END)
    workflow.add_edge("retrieval_node", "critic_node")
    workflow.add_conditional_edges("critic_node", critique_decision)
    workflow.add_edge("rewrite_node", "retrieval_node")
    
    return workflow.compile()

app_graph = build_graph()