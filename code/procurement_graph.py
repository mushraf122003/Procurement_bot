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
from code.mysql_lookup import parse_table_names, run_mysql_lookup
from code.retrieve import retrieve


SYSTEM_PROMPT = """You are a helpful procurement assistant.
Use the supplier_contract tool for supplier contracts, contract clauses, vendor terms,
pricing terms, payment terms, and procurement policy content indexed in MongoDB Atlas.
Use the purchase_order tool only for purchase-order data in MySQL. Use the invoice tool
only for invoice data in MySQL. For document-specific or database-specific claims, use
the relevant tool and rely on the returned context. State when retrieved information is
insufficient. You may answer simple conversational questions without calling a tool."""


class ProcurementState(MessagesState):
    """Conversation state required by the procurement RAG workflow.

    ``messages`` is inherited from ``MessagesState`` and is merged by LangGraph's
    message reducer. The extra fields retain retrieval details for the current
    thread, which makes the workflow easier to inspect, debug, and extend.
    """

    current_question: NotRequired[str]
    retrieved_context: NotRequired[str]
    tool_context: NotRequired[dict[str, str]]
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

    @tool("supplier_contract")
    def supplier_contract(query: str, limit: int = 4) -> str:
        """Supplier Contract: search MongoDB Atlas for supplier contract and policy details."""
        # The tool result is added to the message history before Gemini is called again.
        return format_context(retrieve(query, limit, settings))

    @tool("purchase_order")
    def purchase_order(query: str) -> str:
        """Purchase Order: query configured MySQL purchase-order tables for factual details."""
        if not settings.mysql_uri:
            return "Purchase Order tool is not configured: set MYSQL_URI."
        try:
            return run_mysql_lookup(
                question=query,
                mysql_uri=settings.mysql_uri,
                table_names=parse_table_names(
                    settings.mysql_purchase_order_tables,
                    "MYSQL_PURCHASE_ORDER_TABLES",
                ),
                domain="purchase order",
            )
        except Exception as error:
            return f"Purchase Order lookup failed: {error}"

    @tool("invoice")
    def invoice(query: str) -> str:
        """Invoice: query configured MySQL invoice tables for factual details."""
        if not settings.mysql_uri:
            return "Invoice tool is not configured: set MYSQL_URI."
        try:
            return run_mysql_lookup(
                question=query,
                mysql_uri=settings.mysql_uri,
                table_names=parse_table_names(
                    settings.mysql_invoice_tables,
                    "MYSQL_INVOICE_TABLES",
                ),
                domain="invoice",
            )
        except Exception as error:
            return f"Invoice lookup failed: {error}"

    tools = [supplier_contract, purchase_order, invoice]
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
        tool_messages = [
            message
            for message in state["messages"]
            if isinstance(message, ToolMessage)
            and message.name in {"supplier_contract", "purchase_order", "invoice"}
        ]
        latest_context = str(tool_messages[-1].content) if tool_messages else ""
        tool_context = {
            message.name: str(message.content)
            for message in tool_messages
        }
        # This metadata is checkpointed with messages and survives later turns.
        return {
            "retrieved_context": latest_context,
            "tool_context": tool_context,
            "retrieval_count": len(tool_messages),
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
    return result["messages"][-1].content[0]["text"]


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
