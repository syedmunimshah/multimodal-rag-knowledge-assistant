from typing import TypedDict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.config import settings
from src.prompts import ANSWER_PROMPT, PLAN_PROMPT, SUFFICIENCY_PROMPT, format_context
from src.vectorstore import get_vectorstore


class State(TypedDict):
    question: str
    sub_queries: list[str]
    docs: list[Document]
    hops: int
    answer: str


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.chat_model, api_key=settings.openai_api_key, temperature=0)


def plan_node(state: State) -> dict:
    """Break the question into sub-queries so retrieval covers every part
    of a multi-part question, not just the top-level phrasing."""
    prompt = PLAN_PROMPT.format(max_hops=settings.max_reasoning_hops, question=state["question"])
    response = _llm().invoke(prompt).content
    sub_queries = [line.strip("- ").strip() for line in response.splitlines() if line.strip()]
    return {"sub_queries": sub_queries[: settings.max_reasoning_hops], "hops": 0}


def retrieve_node(state: State) -> dict:
    store = get_vectorstore()
    new_docs: list[Document] = []
    for query in state["sub_queries"]:
        new_docs.extend(store.similarity_search(query, k=settings.top_k))
    return {"docs": state["docs"] + new_docs, "hops": state["hops"] + 1}


def grade_node(state: State) -> dict:
    """Ask the model whether retrieved context is enough to answer, or
    another reasoning hop is needed. Caps at max_reasoning_hops either way."""
    context = format_context(state["docs"])
    prompt = SUFFICIENCY_PROMPT.format(question=state["question"], context=context)
    verdict = _llm().invoke(prompt).content.strip().upper()
    return {"docs": state["docs"], "_sufficient": "SUFFICIENT" in verdict}


def route_after_grade(state: State) -> str:
    if state.get("_sufficient") or state["hops"] >= settings.max_reasoning_hops:
        return "answer"
    return "refine"


def refine_node(state: State) -> dict:
    """Re-plan sub-queries given what's still missing, instead of repeating
    the same retrieval."""
    context = format_context(state["docs"])
    prompt = (
        f"The question is: {state['question']}\n\n"
        f"Context gathered so far:\n{context}\n\n"
        "What is still missing? Write 1-2 follow-up sub-queries, one per line."
    )
    response = _llm().invoke(prompt).content
    sub_queries = [line.strip("- ").strip() for line in response.splitlines() if line.strip()]
    return {"sub_queries": sub_queries}


def answer_node(state: State) -> dict:
    context = format_context(state["docs"])
    prompt = ANSWER_PROMPT.format(context=context, question=state["question"])
    answer = _llm().invoke(prompt).content
    return {"answer": answer}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("refine", refine_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"answer": "answer", "refine": "refine"})
    graph.add_edge("refine", "retrieve")
    graph.add_edge("answer", END)

    return graph.compile()


def ask(question: str) -> str:
    app = build_graph()
    result = app.invoke({"question": question, "sub_queries": [], "docs": [], "hops": 0, "answer": ""})
    return result["answer"]
