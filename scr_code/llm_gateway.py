"""LiteLLM gateway for cached, routed, and fault-tolerant model calls."""

from __future__ import annotations

import os
from typing import Any, Literal, cast

import litellm
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM, ChatLiteLLMRouter
from litellm import Cache, Router


TaskType = Literal["general", "code", "RAG"]
_VALID_TASK_TYPES = {"general", "code", "RAG"}
_CLASSIFIER_MODEL = "gemini/gemini-3.7-flash"
_ROUTER: Router | None = None

# Load local credentials before LiteLLM reads them into the model deployments.
load_dotenv()

# Each duplicate model_name is a deployment in the same LiteLLM model group.
# The router learns deployment latency and selects the fastest healthy option.
MODEL_LIST: list[dict[str, Any]] = [
    {
        "model_name": "code",
        "litellm_params": {
            "model": "groq/openai/gpt-oss-120b",
            "api_key": os.getenv("GROQ_API_KEY"),
        },
        "model_info": {"id": "model:groq/openai/gpt-oss-120b"},
    },
    {
        "model_name": "code",
        "litellm_params": {
            "model": "groq/openai/gpt-oss-20b",
            "api_key": os.getenv("GROQ_API_KEY"),
        },
        "model_info": {"id": "model:groq/openai/gpt-oss-20b"},
    },
    {
        "model_name": "general",
        "litellm_params": {
            "model": "groq/qwen/qwen3.6-27b",
            "api_key": os.getenv("GROQ_API_KEY"),
        },
        "model_info": {"id": "model:groq/qwen/qwen3.6-27b"},
    },
    {
        "model_name": "general",
        "litellm_params": {
            "model": "groq/qwen/qwen3.8-27b",
            "api_key": os.getenv("GROQ_API_KEY"),
        },
        "model_info": {"id": "model:groq/qwen/qwen3.8-27b"},
    },
    {
        "model_name": "RAG",
        "litellm_params": {
            "model": "gemini/gemini-3.7-flash",
            "api_key": os.getenv("GOOGLE_API_KEY"),
        },
        "model_info": {"id": "model:gemini/gemini-3.7-flash"},
    },
    {
        "model_name": "RAG",
        "litellm_params": {
            "model": "gemini/gemini-3.6-flash",
            "api_key": os.getenv("GOOGLE_API_KEY"),
        },
        "model_info": {"id": "model:gemini/gemini-3.6-flash"},
    },
]


def get_litellm_router() -> Router:
    """Return the shared router so latency and local-cache state survive requests."""
    global _ROUTER
    if _ROUTER is None:
        # Local cache avoids repeated provider calls for identical model requests.
        litellm.cache = Cache(type="local")
        _ROUTER = Router(
            model_list=MODEL_LIST,
            routing_strategy="latency-based-routing",
            cache_responses=True,
            num_retries=1,
            # A failed deployment is cooled down immediately so LiteLLM can
            # select the remaining deployment in the requested model group.
            allowed_fails=1,
            max_fallbacks=1,
        )
    return _ROUTER


def get_routed_llm(task_type: TaskType) -> ChatLiteLLMRouter:
    """Create a LangChain model that sends a task to its LiteLLM model group."""
    return ChatLiteLLMRouter(
        router=get_litellm_router(),
        model_name=task_type,
        temperature=0,
        model_kwargs={"caching": True},
    )


def _content_as_text(content: object) -> str:
    """Normalize text or structured LangChain message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ]
        return "".join(text_parts)
    return str(content)


def get_response_text(message: Any) -> str:
    """Extract an LLM response text without assuming provider-specific content shape."""
    return _content_as_text(message.content)


def classify_task(question: str) -> TaskType:
    """Use Gemini Flash to select the general, code, or RAG model group."""
    classifier = ChatLiteLLM(
        model=_CLASSIFIER_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        model_kwargs={"caching": True},
    )
    classifier_prompt = f"""Classify the user request into exactly one label: general, code, or RAG.
general: conversation, explanations, non-tool calling or non-database procurement questions.
code: SQL generation, database-query construction, schemas, or code requests.
RAG: supplier contracts, procurement documents, purchase orders, invoices, or questions
that require retrieving information from the connected procurement databases.

User request: {question}

Return only the label."""
    try:
        label = get_response_text(classifier.invoke(classifier_prompt)).strip().lower()
    except Exception:
        # A classifier outage must not stop the procurement agent from serving a user.
        return "RAG"

    if label == "rag":
        return "RAG"
    if label in _VALID_TASK_TYPES:
        return cast(TaskType, label)
    return "RAG"
