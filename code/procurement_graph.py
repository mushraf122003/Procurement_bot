"""LangGraph workflow that lets Gemini retrieve procurement context when needed."""

from __future__ import annotations

import argparse
from typing import Any, NotRequired

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from code.config import Settings
from code.google_llm import get_google_llm
from code.retrieve import retrieve


SYSTEM_PROMPT = """You are a helpful procurement assistant.
Use the retrieve_procurement_context tool for questions about procurement documents,
vendors, pricing, payment terms, contracts, purchase orders, or policies. Use only
retrieved context for document-specific claims. State when the retrieved context is
insufficient, and cite source file names in your answer when they are available.
You may answer simple conversational questions without calling the tool."""


class ProcurementState(MessagesState):
    """Conversation state required by the procurement RAG workflow.

    ``messages`` is inherited from ``MessagesState`` and is merged by LangGraph's
    message reducer. The extra fields retain retrieval details for the current
    thread, which makes the workflow easier to inspect, debug, and extend.
    """

    current_question: NotRequired[str]
    retrieved_context: NotRequired[str]
    retrieval_count: NotRequired[int]


# MemorySaver stores a separate checkpoint history for each configured thread_id.
# Keep one instance for the running application so later requests reuse its memory.
CHECKPOINTER = MemorySaver()
_COMPILED_GRAPH = None


def format_context(results: list[dict[str, Any]]) -> str:
    """Convert Atlas matches into readable tool context for the Gemini model."""
    if not results:
        return "No relevant procurement document chunks were found."

    sections = []
    for number, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        source = metadata.get("file_name") or metadata.get("source", "Unknown source")
        sections.append(
            f"Source {number}: {source} (score: {result['score']:.3f})\n{result['text']}"
        )
    return "\n\n".join(sections)


def build_procurement_graph(checkpointer: MemorySaver = CHECKPOINTER):
    """Build START -> llm_with_tool -> tools -> llm_with_tool -> END."""
    settings = Settings.from_environment()

    @tool("retrieve_procurement_context")
    def retrieve_procurement_context(query: str, limit: int = 4) -> str:
        """Search indexed procurement files for context relevant to a user question."""
        # The tool result is added to the message history before Gemini is called again.
        return format_context(retrieve(query, limit, settings))

    tools = [retrieve_procurement_context]
    llm_with_tools = get_google_llm().bind_tools(tools)

    def call_model(state: ProcurementState) -> dict[str, Any]:
        # Gemini decides whether to answer immediately or emits a tool call.
        response = llm_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        latest_question = next(
            (
                str(message.content)
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            ),
            state.get("current_question", ""),
        )
        return {"messages": [response], "current_question": latest_question}

    def update_retrieval_state(state: ProcurementState) -> dict[str, Any]:
        """Copy the latest tool output into project-specific state fields."""
        latest_context = next(
            (
                str(message.content)
                for message in reversed(state["messages"])
                if isinstance(message, ToolMessage)
                and message.name == "retrieve_procurement_context"
            ),
            "",
        )
        # This metadata is checkpointed with messages and survives later turns.
        return {
            "retrieved_context": latest_context,
            "retrieval_count": state.get("retrieval_count", 0) + 1,
        }

    graph = StateGraph(ProcurementState)
    graph.add_node("llm_with_tool", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("update_retrieval_state", update_retrieval_state)
    graph.add_edge(START, "llm_with_tool")
    # Route tool calls to MongoDB; otherwise finish with Gemini's final answer.
    graph.add_conditional_edges(
        "llm_with_tool",
        tools_condition,
        {"tools": "tools", END: END},
    )
    # ToolNode appends context as a ToolMessage before that context is checkpointed.
    graph.add_edge("tools", "update_retrieval_state")
    graph.add_edge("update_retrieval_state", "llm_with_tool")
    return graph.compile(checkpointer=checkpointer)


def get_procurement_graph():
    """Return the singleton compiled graph that shares the application checkpointer."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_procurement_graph()
    return _COMPILED_GRAPH


def ask(question: str, thread_id: str = "default") -> str:
    """Answer one question while retaining history for the supplied thread."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if not thread_id.strip():
        raise ValueError("thread_id cannot be empty.")

    # LangGraph loads and saves the checkpoint identified by this thread ID.
    result = get_procurement_graph().invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Procurement question to answer.")
    parser.add_argument(
        "--thread-id",
        default="default",
        help="Conversation identifier used by the LangGraph checkpointer.",
    )
    args = parser.parse_args()
    print(ask(args.question, args.thread_id))


if __name__ == "__main__":
    main()
