"""Qwen llm model initialization for the procurement bot."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def get_qwen_llm() -> ChatOpenAI:
    """Create the Qwen chat model through Hugging Face + Featherless AI."""

    hf_token = os.environ.get("HF_TOKEN", "")

    if not hf_token:
        raise ValueError("HF_TOKEN must be set before calling the Qwen LLM.")

    return ChatOpenAI(
        model="Qwen/Qwen3.8-27B:featherless-ai",
        api_key=hf_token,
        base_url="https://router.huggingface.co/v1",
        temperature=0,
    )
