"""Plain-text JSON helpers for LLM pipeline stages.

Purpose: keep OpenRouter-compatible JSON prompting and parsing in one place
without relying on provider-specific response_format modes.
Scope: invoke a chat model, extract one JSON object, and validate it with the
caller-supplied Pydantic schema.
Dependencies: LangChain chat messages and Pydantic validation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


async def invoke_json_schema[SchemaT: BaseModel](
    *,
    llm: BaseChatModel,
    prompt_messages: Sequence[Any],
    schema: type[SchemaT],
    instruction: str,
) -> SchemaT:
    """Invoke an LLM and validate the first JSON object in its text response."""

    response = await llm.ainvoke(
        [
            *prompt_messages,
            HumanMessage(
                content=(
                    f"{instruction}\n\n"
                    "Return only one valid JSON object. Do not use markdown, code fences, "
                    "comments, trailing commas, or prose outside the JSON object."
                )
            ),
        ]
    )
    response_text = message_content_to_text(response)
    try:
        payload = extract_json_object(response_text)
    except ValueError as exc:
        raise ValueError(_format_json_parse_failure(response, response_text)) from exc
    return schema.model_validate(payload)


def message_content_to_text(message: object) -> str:
    """Extract text content from a LangChain message-like response."""

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, str):
                fragments.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    fragments.append(text)
        return "\n".join(fragments)
    return str(content)


def extract_json_object(text: str) -> dict[str, object]:
    """Parse the first JSON object embedded in a model response."""

    stripped = text.strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM response did not contain a valid JSON object.")


def _format_json_parse_failure(response: object, response_text: str) -> str:
    """Render a compact raw-response diagnostic for fail-fast JSON errors."""

    preview = response_text.strip().replace("\n", " ")[:600] or "<empty>"
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, Mapping):
        finish_reason = metadata.get("finish_reason")
        model_name = metadata.get("model_name")
        return (
            "LLM response did not contain a valid JSON object. "
            f"model={model_name}; finish_reason={finish_reason}; content_preview={preview!r}."
        )
    return f"LLM response did not contain a valid JSON object. content_preview={preview!r}."


__all__ = ["extract_json_object", "invoke_json_schema", "message_content_to_text"]
