from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    question: str
    route: Optional[str]
    retrieved_contexts: List[Dict[str, Any]]
    sql_query: Optional[str]
    sql_result: Optional[str]
    draft_answer: Optional[str]
    critique_verdict: Optional[str]  # "PASS" ou "RETRY"
    retry_count: int
    final_answer: Optional[str]