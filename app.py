"""Streamlit frontend for the ZAF procurement chatbot.

This file only provides the UI layer. It reuses the existing LangGraph
chatbot entry point: ``ask(question, thread_id)`` from ``scr_code.procurement_graph``.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from time import perf_counter

import streamlit as st

# pyrefly: ignore [missing-import]
from scr_code.procurement_graph import ask


st.set_page_config(
    page_title="ZAF",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


_LIVE_STOPWATCH = st.components.v2.component(
    "zaf_live_stopwatch",
    isolate_styles=False,
    html="""
    <div class="zaf-processing">
        <div class="zaf-wave" aria-hidden="true"><i></i><i></i><i></i><i></i></div>
        <span class="zaf-processing-label" role="status">Generating your response…</span>
        <span class="zaf-timer" role="timer" aria-label="Elapsed minutes and seconds">00:00.0</span>
    </div>
    """,
    js="""
    export default function ({ data, parentElement }) {
        const clock = parentElement.querySelector('[role="timer"]');
        const startedAt = performance.now() - data.elapsed * 1000;
        const update = () => {
            const tenths = Math.floor((performance.now() - startedAt) / 100);
            const minutes = String(Math.floor(tenths / 600)).padStart(2, "0");
            const seconds = String(Math.floor(tenths / 10) % 60).padStart(2, "0");
            clock.textContent = `${minutes}:${seconds}.${tenths % 10}`;
        };
        update();
        const interval = setInterval(update, 50);
        return () => clearInterval(interval);
    }
    """,
)


def apply_jarvis_theme() -> None:
    """Keep the responsive HUD theme inside this file, without external assets."""
    st.html(
        """
        <style>
        :root {
            --zaf-bg: #050b13;
            --zaf-cyan: #64e9ff;
            --zaf-gold: #e6ba73;
            --zaf-text: #e1edf6;
            --zaf-muted: #94afc1;
            --zaf-line: rgba(100, 233, 255, 0.18);
        }
        .stApp {
            background:
                radial-gradient(ellipse at 80% 5%, #102a4080, transparent 48%),
                linear-gradient(rgba(100, 233, 255, 0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(100, 233, 255, 0.025) 1px, transparent 1px),
                var(--zaf-bg);
            background-size: auto, 48px 48px, 48px 48px, auto;
            color: var(--zaf-text);
            color-scheme: dark;
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container {
            max-width: 1080px;
            padding-top: 3.2rem;
            padding-bottom: 3rem;
        }
        .zaf-topline, .zaf-section, .zaf-header-foot {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            font: 0.69rem "Consolas", monospace;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--zaf-muted);
        }
        .zaf-topline { margin-bottom: 1.2rem; }
        .zaf-brand { color: var(--zaf-cyan); }
        .zaf-dot {
            display: inline-block;
            width: 6px;
            height: 6px;
            margin-right: 8px;
            border-radius: 50%;
            background: var(--zaf-cyan);
            box-shadow: 0 0 10px #64e9ff88;
        }
        .zaf-header {
            position: relative;
            padding: 2rem 2.2rem 1rem;
            border: 1px solid var(--zaf-line);
            border-radius: 6px 28px 6px 6px;
            background: linear-gradient(115deg, #0c1b2bf2, #07121ef2);
            box-shadow: 0 18px 65px #00000035, inset 0 1px #b8f4ff09;
            overflow: hidden;
        }
        .zaf-header::before {
            content: "";
            position: absolute;
            left: 0;
            top: 26px;
            width: 3px;
            height: 54px;
            background: var(--zaf-cyan);
            box-shadow: 0 0 18px #64e9ff88;
            pointer-events: none;
        }
        .zaf-hero { display: flex; align-items: center; gap: 2rem; }
        .zaf-intro { flex: 1; min-width: 0; }
        .zaf-kicker {
            color: var(--zaf-cyan);
            font: 0.7rem "Consolas", monospace;
            letter-spacing: 0.2em;
            text-transform: uppercase;
        }
        .zaf-header h1 {
            color: #f1faff;
            font-size: clamp(3.3rem, 8vw, 5rem);
            line-height: 1.15;
            letter-spacing: 0.15em;
            font-weight: 800;
            margin: 0.3rem 0;
            padding: 0;
            text-shadow: 0 0 30px #64e9ff25;
        }
        .zaf-tagline { color: var(--zaf-text); font-size: 1.1rem; }
        .zaf-subtitle {
            color: var(--zaf-muted);
            max-width: 490px;
            margin: 0.5rem 0 1.5rem;
            font-size: 0.92rem;
            line-height: 1.6;
        }
        .zaf-reactor {
            width: 170px;
            height: 170px;
            position: relative;
            display: grid;
            place-items: center;
            flex-shrink: 0;
            margin: 0 1.4rem 1rem;
            border: 1px solid #64e9ff22;
            border-radius: 50%;
            background: radial-gradient(circle, #64e9ff20, transparent 66%);
        }
        .zaf-reactor::before, .zaf-reactor::after {
            content: "";
            position: absolute;
            border-radius: 50%;
            pointer-events: none;
        }
        .zaf-reactor::before {
            inset: 9px;
            border: 3px solid #64e9ff19;
            border-top-color: var(--zaf-cyan);
            border-bottom-color: var(--zaf-cyan);
            animation: zaf-orbit 24s linear infinite;
        }
        .zaf-reactor::after {
            inset: 24px;
            border: 1px dashed #64e9ff88;
            animation: zaf-orbit 36s linear infinite reverse;
        }
        .zaf-reactor-core {
            width: 82px;
            height: 82px;
            display: grid;
            place-items: center;
            border: 1px solid #64e9ff99;
            border-radius: 50%;
            color: #d5faff;
            font: 1.6rem "Consolas", monospace;
            box-shadow: 0 0 28px #64e9ff22, inset 0 0 22px #64e9ff22;
        }
        .zaf-header-foot {
            border-top: 1px solid var(--zaf-line);
            padding-top: 0.9rem;
            font-size: 0.63rem;
        }
        .zaf-capabilities {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.6rem;
        }
        .zaf-capability {
            border: 1px solid #64e9ff14;
            border-radius: 6px;
            background: #0a1622b3;
            padding: 0.9rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.9rem;
            transition: border-color 160ms, background 160ms;
        }
        .zaf-capability:hover { border-color: #64e9ff55; background: #102132; }
        .zaf-capability-number { font: 0.8rem "Consolas", monospace; color: var(--zaf-cyan); }
        .zaf-capability strong { display: block; font-size: 0.87rem; font-weight: 500; }
        .zaf-capability small { color: var(--zaf-muted); font-size: 0.73rem; }
        .zaf-section { margin-bottom: 1.1rem; }
        .zaf-section span:last-child { font-size: 0.62rem; letter-spacing: 0.06em; }
        [data-testid="stChatMessage"] {
            border: 1px solid var(--zaf-line);
            border-left: 2px solid #64e9ff70;
            border-radius: 4px 14px 14px 14px;
            background: #0a1826ed;
            padding: 1.2rem;
            margin-bottom: 0.6rem;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            border: 1px solid #92b0d528;
            border-right: 2px solid var(--zaf-gold);
            border-radius: 14px 4px 14px 14px;
            background: #142234ed;
            margin-left: clamp(0rem, 4vw, 3rem);
        }
        [data-testid="chatAvatarIcon-assistant"] {
            color: var(--zaf-cyan); background: #14394a;
        }
        [data-testid="chatAvatarIcon-user"] { color: var(--zaf-gold); background: #3b3440; }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            color: var(--zaf-text);
            overflow-wrap: anywhere;
            line-height: 1.7;
        }
        [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p {
            color: var(--zaf-muted);
            font: 0.72rem "Consolas", monospace;
        }
        [data-testid="stChatMessage"] a { color: var(--zaf-cyan); }
        [data-testid="stChatMessage"] pre { background: #050e19; border: 1px solid var(--zaf-line); }
        [data-testid="stChatMessage"] code { color: #a8e9f5; background: #050e19; }
        [data-testid="stChatMessage"] th, [data-testid="stChatMessage"] td {
            border-color: #64e9ff25;
        }
        [data-testid="stBottom"], [data-testid="stBottom"] > div {
            background: var(--zaf-bg);
        }
        [data-testid="stBottomBlockContainer"] { max-width: 1080px; padding-bottom: 1.3rem; }
        .st-key-zaf_question [data-testid="stChatInput"] {
            border: 1px solid #64e9ff55;
            border-radius: 10px;
            background: #0d1b2a;
            box-shadow: 0 0 28px #64e9ff08;
        }
        .st-key-zaf_question [data-testid="stChatInput"]:focus-within {
            border-color: var(--zaf-cyan);
            box-shadow: 0 0 0 2px #64e9ff22;
        }
        .st-key-zaf_question textarea { color: var(--zaf-text); caret-color: var(--zaf-cyan); }
        .st-key-zaf_question textarea::placeholder { color: var(--zaf-muted); }
        .st-key-zaf_question button { color: var(--zaf-cyan); }
        .st-key-zaf_question textarea:disabled { -webkit-text-fill-color: var(--zaf-muted); }
        .zaf-processing { display: flex; align-items: center; flex-wrap: wrap; gap: 1rem; }
        .zaf-processing-label { color: var(--zaf-cyan); font-size: 0.88rem; }
        .zaf-timer {
            margin-left: auto;
            font: 1rem "Consolas", monospace;
            font-variant-numeric: tabular-nums;
            color: var(--zaf-cyan);
            white-space: nowrap;
        }
        .zaf-wave { display: flex; align-items: center; gap: 3px; height: 24px; }
        .zaf-wave i {
            display: block;
            width: 3px;
            height: 18px;
            background: var(--zaf-cyan);
            animation: zaf-wave 1.2s ease-in-out infinite;
        }
        .zaf-wave i:nth-child(2) { animation-delay: 0.15s; }
        .zaf-wave i:nth-child(3) { animation-delay: 0.3s; }
        .zaf-wave i:nth-child(4) { animation-delay: 0.45s; }
        @keyframes zaf-orbit { to { transform: rotate(360deg); } }
        @keyframes zaf-wave { 0%, 100% { transform: scaleY(0.3); opacity: 0.5; } 50% { transform: scaleY(1); opacity: 1; } }
        @media (max-width: 640px) {
            .block-container { padding: 2rem 1rem 2rem; }
            .zaf-header { padding: 1.4rem 1.2rem 0.9rem; }
            .zaf-hero { gap: 0.5rem; }
            .zaf-reactor { width: 90px; height: 90px; margin: 0; }
            .zaf-reactor::before { inset: 5px; border-width: 2px; }
            .zaf-reactor::after { inset: 14px; }
            .zaf-reactor-core { width: 44px; height: 44px; font-size: 1rem; }
            .zaf-capabilities { gap: 0.5rem; }
            .zaf-capability { padding: 0.7rem; gap: 0; }
            .zaf-capability-number, .zaf-capability small { display: none; }
            .zaf-capability strong { font-size: 0.76rem; }
            .zaf-topline, .zaf-header-foot { letter-spacing: 0.04em; font-size: 0.58rem; }
            .zaf-section span:last-child { display: none; }
            [data-testid="stChatMessage"] { padding: 0.85rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            .zaf-reactor::before, .zaf-reactor::after, .zaf-wave i { animation: none; }
            .zaf-capability { transition: none; }
        }
        </style>
        """
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
    st.session_state.setdefault("pending_response", None)


def render_header() -> None:
    """Render decorative HUD graphics and capabilities, not connection claims."""
    st.html(
        """
        <div class="zaf-topline">
            <span class="zaf-brand">ZAF / PROCUREMENT INTELLIGENCE</span>
            <span><span class="zaf-dot"></span>CONVERSATION WORKSPACE</span>
        </div>
        <header class="zaf-header">
            <div class="zaf-hero">
                <div class="zaf-intro">
                    <div class="zaf-kicker">Your procurement copilot</div>
                    <h1>ZAF</h1>
                    <div class="zaf-tagline">Clarity at your command.</div>
                    <p class="zaf-subtitle">Explore your Procurement data. One conversation. Connected context.</p>
                </div>
                <div class="zaf-reactor" aria-hidden="true">
                    <div class="zaf-reactor-core">Z</div>
                </div>
            </div>
            <div class="zaf-header-foot">
                <span>LANGGRAPH / SESSION MEMORY</span>
                <span>CONTEXT → REASONING → RESPONSE</span>
            </div>
        </header>
        <div class="zaf-capabilities">
            <div class="zaf-capability"><span class="zaf-capability-number">01 /</span>
                <div><strong>Supplier contracts</strong><small>MongoDB · Vector retrieval</small></div></div>
            <div class="zaf-capability"><span class="zaf-capability-number">02 /</span>
                <div><strong>Purchase orders</strong><small>MySQL · Order lookup</small></div></div>
            <div class="zaf-capability"><span class="zaf-capability-number">03 /</span>
                <div><strong>Invoices</strong><small>MySQL · Invoice lookup</small></div></div>
        </div>
        <div class="zaf-section">
            <span>01 / Conversation</span>
            <span>Ask naturally. Follow up with context.</span>
        </div>
        """
    )


def render_chat_history() -> None:
    """Display the conversation and each answer's original measured duration."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.caption("YOU" if message["role"] == "user" else "ZAF / ASSISTANT")
            st.markdown(message["content"])
            if message.get("response_time") is not None:
                status = " · Request failed" if message.get("failed") else ""
                st.caption(
                    f"⏱ Response Time: {message['response_time']:.2f} seconds{status}"
                )


def generate_response(question: str, thread_id: str, started_at: float) -> dict:
    """Call the unchanged backend off the UI thread and time actual completion.

    No Streamlit APIs are used in this worker. Timing includes dispatch and the
    complete ask() call, including any retrieval, routing, and model retries.
    """
    failed = False
    try:
        answer = ask(question, thread_id)
    except Exception as error:
        failed = True
        answer = (
            "I could not complete that request. Please check your LLM provider, "
            f"MongoDB, and MySQL configuration. Technical detail: {error}"
        )
    elapsed = perf_counter() - started_at
    return {
        "role": "assistant",
        "content": answer,
        "response_time": elapsed,
        "failed": failed,
    }


def handle_user_question() -> None:
    """Start timing in the submit callback, before the page redraws.

    Keep the future in this browser session so reruns cannot lose the response
    or start a duplicate backend request for the same conversation turn.
    """
    started_at = perf_counter()
    question = st.session_state.get("zaf_question", "")
    if not question.strip() or st.session_state.pending_response is not None:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zaf-response")
    try:
        future = executor.submit(
            copy_context().run,
            generate_response,
            question,
            st.session_state.thread_id,
            started_at,
        )
        st.session_state.pending_response = {
            "future": future,
            "started_at": started_at,
        }
    finally:
        executor.shutdown(wait=False)


def render_pending_response() -> None:
    """Let the browser tick independently while waiting for the existing worker.

    The stopwatch uses a monotonic browser clock, seeded with actual server
    elapsed time. Only completion reruns the app; individual ticks do not.
    """
    pending = st.session_state.pending_response
    if pending is None:
        return

    if not pending["future"].done():
        with st.chat_message("assistant"):
            st.caption("ZAF / ASSISTANT")
            _LIVE_STOPWATCH(
                data={"elapsed": perf_counter() - pending["started_at"]},
                key="zaf_stopwatch",
            )
            st.caption("Stopwatch · Running until the complete answer is ready")

    st.session_state.messages.append(pending["future"].result())
    st.session_state.pending_response = None
    st.rerun()


def main() -> None:
    """Run the Streamlit chatbot frontend."""
    apply_jarvis_theme()
    initialize_session()
    render_header()
    render_chat_history()

    st.chat_input(
        "Ask ZAF about contracts, purchase orders, or invoices…",
        key="zaf_question",
        disabled=st.session_state.pending_response is not None,
        on_submit=handle_user_question,
    )
    if st.session_state.pending_response is not None:
        render_pending_response()


if __name__ == "__main__":
    main()
