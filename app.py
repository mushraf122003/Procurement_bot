"""Streamlit frontend for the ZAF procurement chatbot.

This file only provides the UI layer. It reuses the existing LangGraph
chatbot entry point: ``ask(question, thread_id)`` from ``scr_code.procurement_graph``.
"""

from __future__ import annotations

import uuid

import streamlit as st

# pyrefly: ignore [missing-import]
from scr_code.procurement_graph import ask


st.set_page_config(
    page_title="ZAF",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_jarvis_theme() -> None:
    """Apply the Iron Man / JARVIS-inspired HUD styling to the Streamlit app."""
    st.markdown(
        """
        <style>
        :root {
            --zaf-bg: #050914;
            --zaf-panel: rgba(7, 21, 40, 0.78);
            --zaf-cyan: #00e5ff;
            --zaf-blue: #3b82f6;
            --zaf-gold: #f8c14a;
            --zaf-text: #e6f7ff;
            --zaf-muted: #8bb7cc;
        }

        .stApp {
            background:
                radial-gradient(circle at 20% 15%, rgba(0, 229, 255, 0.16), transparent 28%),
                radial-gradient(circle at 82% 12%, rgba(59, 130, 246, 0.16), transparent 26%),
                linear-gradient(135deg, #030712 0%, #061526 52%, #020617 100%);
            color: var(--zaf-text);
        }

        .block-container {
            max-width: 980px;
            padding-top: 2.2rem;
            padding-bottom: 6.5rem;
        }

        .zaf-header {
            position: relative;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.4rem;
            border: 1px solid rgba(0, 229, 255, 0.35);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(8, 30, 55, 0.82), rgba(2, 8, 23, 0.72));
            box-shadow:
                0 0 32px rgba(0, 229, 255, 0.12),
                inset 0 0 28px rgba(0, 229, 255, 0.05);
            overflow: hidden;
        }

        .zaf-header::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, transparent 0%, rgba(0, 229, 255, 0.14) 50%, transparent 100%),
                repeating-linear-gradient(90deg, rgba(255,255,255,0.04) 0 1px, transparent 1px 58px);
            pointer-events: none;
        }

        .zaf-kicker {
            position: relative;
            color: var(--zaf-gold);
            font-size: 0.78rem;
            letter-spacing: 0.22rem;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }

        .zaf-title {
            position: relative;
            color: var(--zaf-text);
            font-size: clamp(2.2rem, 6vw, 4.6rem);
            line-height: 0.95;
            letter-spacing: 0.18rem;
            font-weight: 800;
            text-shadow: 0 0 22px rgba(0, 229, 255, 0.32);
        }

        .zaf-subtitle {
            position: relative;
            color: var(--zaf-muted);
            max-width: 720px;
            margin-top: 0.65rem;
            font-size: 1rem;
        }

        .zaf-status {
            display: flex;
            gap: 0.65rem;
            flex-wrap: wrap;
            margin: 0 0 1.25rem 0;
        }

        .zaf-chip {
            border: 1px solid rgba(0, 229, 255, 0.28);
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            color: #bff7ff;
            background: rgba(2, 10, 24, 0.58);
            box-shadow: inset 0 0 14px rgba(0, 229, 255, 0.06);
            font-size: 0.84rem;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid rgba(0, 229, 255, 0.18);
            border-radius: 16px;
            background: rgba(4, 14, 30, 0.68);
            box-shadow: 0 0 20px rgba(0, 229, 255, 0.06);
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            border-color: rgba(248, 193, 74, 0.28);
            background: rgba(29, 20, 8, 0.52);
        }

        [data-testid="stChatInput"] {
            border: 1px solid rgba(0, 229, 255, 0.35);
            border-radius: 18px;
            background: rgba(3, 11, 24, 0.92);
            box-shadow: 0 0 24px rgba(0, 229, 255, 0.14);
        }

        .stChatInput textarea {
            color: #00FFFF !important;
            background-color: #111111 !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #888888 !important;
        }

        .stMarkdown, .stMarkdown p {
            color: var(--zaf-text);
        }

        hr {
            border-color: rgba(0, 229, 255, 0.18);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session() -> None:
    """Create persistent Streamlit session state for one browser chat session."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "ZAF online. Ask me about supplier contracts, purchase orders, or invoices.",
            }
        ]


def render_header() -> None:
    """Render the application title and lightweight HUD labels."""
    st.markdown(
        """
        <div class="zaf-header">
            <div class="zaf-kicker">Procurement Intelligence Interface</div>
            <div class="zaf-title">ZAF</div>
            <div class="zaf-subtitle">
                A JARVIS-inspired command console for contract retrieval,
                purchase order lookup, invoice analysis, and grounded answers.
            </div>
        </div>
        <div class="zaf-status">
            <span class="zaf-chip">LangGraph Memory Active</span>
            <span class="zaf-chip">MongoDB Vector Search</span>
            <span class="zaf-chip">MySQL Procurement Tools</span>
            <span class="zaf-chip">Gemini Reasoning Core</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    """Display messages already stored in Streamlit session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_question(question: str) -> None:
    """Send the user question to the existing backend and append the answer."""
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("ZAF is scanning the procurement systems..."):
            try:
                answer = ask(question, st.session_state.thread_id)
            except Exception as error:
                answer = (
                    "I could not complete that request. Please check your Google, "
                    f"MongoDB, and MySQL configuration. Technical detail: {error}"
                )
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    """Run the Streamlit chatbot frontend."""
    apply_jarvis_theme()
    initialize_session()
    render_header()
    render_chat_history()

    question = st.chat_input("Ask ZAF about contracts, purchase orders, or invoices...")
    if question:
        handle_user_question(question)


if __name__ == "__main__":
    main()
