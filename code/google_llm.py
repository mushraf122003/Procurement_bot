"""Google Gemini model initialization for the procurement bot."""

from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI


def get_google_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model used by the LangGraph answer node."""
    # The API key remains in the environment instead of being stored in source code.
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY must be set before calling the Google LLM.")

    # A low temperature keeps procurement answers consistent and evidence-focused.
    return ChatGoogleGenerativeAI(
        model=os.environ.get("GOOGLE_MODEL", "gemini-3.6-flash"),
        google_api_key=google_api_key,
        temperature=0.5,
    )
