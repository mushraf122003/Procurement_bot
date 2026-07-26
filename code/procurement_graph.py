"""LangGraph workflow that lets Gemini retrieve procurement context when needed."""

from __future__ import annotations

import argparse
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
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


def build_procurement_graph():
    """Build START -> llm_with_tool -> tools -> llm_with_tool -> END."""
    settings = Settings.from_environment()

    @tool("retrieve_procurement_context")
    def retrieve_procurement_context(query: str, limit: int = 4) -> str:
        """Search indexed procurement files for context relevant to a user question."""
        # The tool result is added to the message history before Gemini is called again.
        return format_context(retrieve(query, limit, settings))

    tools = [retrieve_procurement_context]
    llm_with_tools = get_google_llm().bind_tools(tools)

    def call_model(state: MessagesState) -> dict[str, list]:
        # Gemini decides whether to answer immediately or emits a tool call.
        response = llm_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("llm_with_tool", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "llm_with_tool")
    # Route tool calls to MongoDB; otherwise finish with Gemini's final answer.
    graph.add_conditional_edges(
        "llm_with_tool",
        tools_condition,
        {"tools": "tools", END: END},
    )
    # ToolNode appends context as a ToolMessage, then Gemini evaluates it on the next turn.
    graph.add_edge("tools", "llm_with_tool")
    return graph.compile()


def ask(question: str) -> str:
    """Run one user question through the procurement LangGraph workflow."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    result = build_procurement_graph().invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Procurement question to answer.")
    args = parser.parse_args()
    print(ask(args.question))


if __name__ == "__main__":
    main()
