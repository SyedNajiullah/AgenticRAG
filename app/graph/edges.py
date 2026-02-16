from langgraph.graph import StateGraph, START, END
from ..llms.models import ChatState
from .nodes import planning, context_search, check_relevence, web_search, fall_back_llm
from ..utils import return_phase


def create_graph():
    graph = StateGraph(ChatState)

    graph.add_node('planning', planning)
    graph.add_node('context_search', context_search)
    graph.add_node('check_relevence', check_relevence)
    graph.add_node('web_search', web_search)
    graph.add_node("fall_back_llm", fall_back_llm)

    graph.add_edge(START, "planning")
    graph.add_conditional_edges('planning', return_phase, {"fall_back_llm": "fall_back_llm", "rag": "context_search"})
    graph.add_edge('context_search', "check_relevence")
    graph.add_conditional_edges("check_relevence", return_phase, {"web": "web_search", "end": END, "fall_back_llm": "fall_back_llm"})
    graph.add_edge('web_search', "check_relevence")
    graph.add_edge("fall_back_llm", END)

    return graph

create_graph()