from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import uuid
from typing import Any, Iterable, Sequence


_TOOL_WORD = re.compile(r"[a-z0-9]+")
_TOOL_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "tool",
    "tools",
    "use",
    "using",
    "reply",
    "exactly",
    "do",
    "not",
    "this",
    "that",
    "please",
    "can",
    "will",
    "with",
}
_CORE_AGENT_TOOLS = {
    "bash",
    "edit",
    "editfile",
    "read",
    "readfile",
    "write",
    "writefile",
}


def responses_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = str(payload.get("instructions") or "").strip()
    if instructions:
        messages.append({"role": "system", "content": instructions})

    raw_input = payload.get("input", "")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
        return messages
    if not isinstance(raw_input, list):
        raise ValueError("Responses input must be text or an input item list")

    for item in raw_input:
        if not isinstance(item, dict):
            raise ValueError("Responses input items must be objects")
        item_type = str(item.get("type") or "message")
        if item_type == "message":
            messages.append(
                {
                    "role": str(item.get("role") or "user"),
                    "content": _responses_content(item.get("content", "")),
                }
            )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or item.get("id") or ""),
                    "content": str(item.get("output") or ""),
                }
            )
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": str(item.get("call_id") or item.get("id") or ""),
                            "type": "function",
                            "function": {
                                "name": str(item.get("name") or ""),
                                "arguments": str(item.get("arguments") or "{}"),
                            },
                        }
                    ],
                }
            )
        else:
            raise ValueError(f"unsupported Responses input item: {item_type}")
    return messages


def responses_tools(tools: Any) -> list[dict[str, Any]]:
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ValueError("Responses tools must be a list")
    result = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        result.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool.get("name") or ""),
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return result


def anthropic_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = payload.get("system")
    if system:
        messages.append({"role": "system", "content": _anthropic_text(system)})

    raw_messages = payload.get("messages") or []
    if not isinstance(raw_messages, list):
        raise ValueError("Anthropic messages must be a list")
    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise ValueError("Anthropic messages must be objects")
        role = str(raw.get("role") or "user")
        content = raw.get("content", "")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
        text_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError("Anthropic content blocks must be objects")
            part_type = str(part.get("type") or "text")
            if part_type == "text":
                text_parts.append({"type": "text", "text": str(part.get("text") or "")})
            elif part_type == "image":
                text_parts.append(_anthropic_image_part(part))
            elif part_type == "tool_use":
                tool_calls.append(
                    {
                        "id": str(part.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": str(part.get("name") or ""),
                            "arguments": json.dumps(
                                part.get("input") or {}, separators=(",", ":")
                            ),
                        },
                    }
                )
            elif part_type == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(part.get("tool_use_id") or ""),
                        "content": _anthropic_text(part.get("content", "")),
                    }
                )
            elif part_type in {"thinking", "redacted_thinking"}:
                # Thinking blocks are transport metadata, not conversation context.
                continue
            else:
                raise ValueError(f"unsupported Anthropic content block: {part_type}")

        if text_parts or tool_calls:
            message: dict[str, Any] = {"role": role, "content": text_parts}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
        messages.extend(tool_results)
    return messages


def anthropic_tools(tools: Any) -> list[dict[str, Any]]:
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ValueError("Anthropic tools must be a list")
    return [
        {
            "type": "function",
            "function": {
                "name": str(tool.get("name") or ""),
                "description": str(tool.get("description") or ""),
                "parameters": tool.get("input_schema")
                or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
        if isinstance(tool, dict)
    ]


def select_anthropic_tools(
    tools: Any,
    messages: Any,
    *,
    tool_choice: Any = None,
    limit: int = 48,
) -> list[dict[str, Any]]:
    """Keep the useful agent tools without prefilling every connected schema."""
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ValueError("Anthropic tools must be a list")
    candidates = [tool for tool in tools if isinstance(tool, dict)]
    if limit <= 0:
        return candidates

    query = _initial_anthropic_user_text(messages)
    query_terms = _tool_terms(query)
    used_names = _anthropic_tool_names(messages)
    forced_name = ""
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        forced_name = str(tool_choice.get("name") or "")

    ranked: list[tuple[int, int, int, dict[str, Any]]] = []
    mandatory: set[int] = set()
    for index, tool in enumerate(candidates):
        name = str(tool.get("name") or "")
        leaf_name = name.rsplit("__", 1)[-1]
        normalized_name = "".join(_TOOL_WORD.findall(leaf_name.lower()))
        name_terms = _tool_terms(name.replace("_", " "))
        description_terms = _tool_terms(str(tool.get("description") or ""))
        schema = tool.get("input_schema")
        property_terms = _schema_property_terms(schema)
        score = 0
        if name == forced_name:
            score += 100_000
            mandatory.add(index)
        if name in used_names:
            score += 50_000
            mandatory.add(index)
        if normalized_name in _CORE_AGENT_TOOLS:
            score += 20_000
            mandatory.add(index)
        score += 240 * len(query_terms & name_terms)
        score += 35 * len(query_terms & property_terms)
        score += 12 * len(query_terms & description_terms)
        if normalized_name and normalized_name in "".join(sorted(query_terms)):
            score += 300
        encoded_size = len(json.dumps(tool, separators=(",", ":"), ensure_ascii=True))
        ranked.append((score, encoded_size, index, tool))

    selected_indexes = set(mandatory)
    minimum = min(limit, 4)
    for score, _, index, _ in sorted(
        ranked,
        key=lambda row: (-row[0], row[1], row[2]),
    ):
        if len(selected_indexes) >= max(limit, len(mandatory)):
            break
        # A single generic description word ("context", "answer", "task")
        # should not drag a multi-kilobyte schema into every Claude request.
        if score < 100 and len(selected_indexes) >= minimum:
            continue
        selected_indexes.add(index)
    return [tool for index, tool in enumerate(candidates) if index in selected_indexes]


def anthropic_cache_affinity(
    payload: dict[str, Any],
    selected_tools: Sequence[dict[str, Any]],
) -> str:
    """Return a stable cache lane for one Claude conversation.

    Claude Code sends small helper requests beside its large coding request. They
    share credentials and a process, but not a useful prompt prefix. Keeping a
    separate utility lane prevents title/status calls from replacing the coding
    conversation's native KV state.
    """
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    identity: Any = metadata.get("session_id") or metadata.get("conversation_id")
    raw_user_id = metadata.get("user_id")
    if not identity and isinstance(raw_user_id, str):
        try:
            decoded = json.loads(raw_user_id)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, dict):
            identity = decoded.get("session_id") or decoded.get("conversation_id")
    if not identity:
        identity = {
            "system": payload.get("system"),
            "first_user": _initial_anthropic_user_text(payload.get("messages")),
        }
    lane = "agent" if selected_tools else "utility"
    encoded = json.dumps(
        {
            "model": str(payload.get("model") or ""),
            "identity": identity,
            "lane": lane,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return f"anthropic-{lane}-{hashlib.sha256(encoded).hexdigest()[:24]}"


def claude_code_session_title(payload: dict[str, Any]) -> str | None:
    """Answer Claude Code's title helper without occupying an inference replica."""
    system = _anthropic_text(payload.get("system", ""))
    if not (
        "You are naming a coding session" in system
        and 'single "title" field' in system
        and not payload.get("tools")
    ):
        return None

    source = ""
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            source = _anthropic_text(message.get("content", ""))
            match = re.search(r"(?is)<session>(.*?)</session>", source)
            if match:
                source = match.group(1)
            break

    source = re.sub(r"(?is)<[^>]+>", " ", source)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]*", source)
    ignored = {
        "a",
        "an",
        "and",
        "can",
        "exactly",
        "for",
        "me",
        "please",
        "reply",
        "the",
        "to",
        "with",
        "you",
    }
    meaningful = [word for word in words if word.lower() not in ignored]
    title = " ".join(meaningful[:4]).strip(" .,:;-/")
    return title or "Coding session"


def is_claude_code_request(payload: dict[str, Any], user_agent: str = "") -> bool:
    """Recognize Claude Code even when proxies rewrite its User-Agent header."""
    if "claude-cli/" in str(user_agent or "").lower():
        return True
    system = _anthropic_text(payload.get("system", ""))
    return (
        "You are an interactive agent that helps users with software engineering tasks."
        in system
    )


def compact_claude_code_tools(
    tools: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep callable tool contracts while removing client-side help prose."""
    compacted: list[dict[str, Any]] = []
    for tool in tools:
        name = str(tool.get("name") or "")
        compacted.append(
            {
                "name": name,
                "description": _concise_tool_description(
                    str(tool.get("description") or ""),
                    fallback=f"Call {name}.",
                ),
                "input_schema": _compact_json_schema(
                    tool.get("input_schema")
                    or {"type": "object", "properties": {}}
                ),
            }
        )
    return compacted


def _concise_tool_description(text: str, *, fallback: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return fallback
    sentence = re.match(r"^(.+?[.!?])(?:\s|$)", normalized)
    concise = sentence.group(1) if sentence else normalized
    if len(concise) <= 160:
        return concise
    return concise[:157].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."


def _compact_json_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_compact_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    prose_keys = {
        "$comment",
        "default",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
    }
    return {
        key: _compact_json_schema(item)
        for key, item in value.items()
        if key not in prose_keys
    }


def compact_claude_code_messages(
    messages: Sequence[dict[str, Any]],
    *,
    selected_tool_names: set[str],
) -> list[dict[str, Any]]:
    """Remove Claude Code harness text that duplicates the selected tool schemas."""
    compacted: list[dict[str, Any]] = []
    for message in messages:
        if str(message.get("role") or "") != "system":
            compacted.append(message)
            continue
        text = _anthropic_text(message.get("content", "")).strip()
        if (
            "You are an interactive agent that helps users with software engineering tasks."
            in text
        ):
            compacted.append(
                {
                    **message,
                    "content": _compact_claude_code_system(text),
                }
            )
            continue
        if text.startswith("Available agent types for the Agent tool:"):
            catalog = _compact_claude_code_catalog(text, selected_tool_names)
            if catalog:
                compacted.append({**message, "content": catalog})
            continue
        compacted.append(message)
    return compacted


def _compact_claude_code_system(text: str) -> str:
    environment = ""
    match = re.search(r"(?ms)^# Environment\s*\n(.*?)(?=^#\s|\Z)", text)
    if match:
        allowed = (
            "Primary working directory:",
            "Is a git repository:",
            "Platform:",
            "Shell:",
            "OS Version:",
        )
        environment_lines = [
            line.rstrip()
            for line in match.group(1).splitlines()
            if any(field in line for field in allowed)
        ]
        if environment_lines:
            environment = "\n\nEnvironment:\n" + "\n".join(environment_lines)
    return (
        "You are an interactive coding agent inside Claude Code. Use the provided "
        "tools to inspect the repository before editing, match its existing style, "
        "run relevant checks, and report results accurately. Respect denied tool "
        "calls. Confirm destructive or external actions unless the user explicitly "
        "authorized them. Keep user-facing responses concise."
        + environment
    )


def _compact_claude_code_catalog(text: str, selected_tool_names: set[str]) -> str:
    lines: list[str] = []
    if "Agent" in selected_tool_names:
        agent_section = text.split(
            "The following skills are available for use with the Skill tool:", 1
        )[0]
        names = re.findall(r"(?m)^- ([A-Za-z0-9_-]+):", agent_section)
        if names:
            lines.append("Available Agent subtypes: " + ", ".join(names[:16]) + ".")
    if "Skill" in selected_tool_names:
        skill_section = text.split(
            "The following skills are available for use with the Skill tool:", 1
        )[-1]
        names = re.findall(r"(?m)^- ([A-Za-z0-9_-]+):", skill_section)
        if names:
            lines.append("Available Skill names: " + ", ".join(names[:32]) + ".")
    return "\n".join(lines)


def _initial_anthropic_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _anthropic_text(message.get("content", ""))
        text = re.sub(
            r"(?is)<system-reminder>.*?</system-reminder>",
            " ",
            text,
        )
        return text[-16_000:]
    return ""


def _anthropic_tool_names(messages: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(messages, list):
        return names
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                name = str(part.get("name") or "")
                if name:
                    names.add(name)
    return names


def _tool_terms(value: str) -> set[str]:
    return {
        term
        for term in _TOOL_WORD.findall(value.lower())
        if len(term) > 1 and term not in _TOOL_STOP_WORDS
    }


def _schema_property_terms(schema: Any) -> set[str]:
    if not isinstance(schema, dict):
        return set()
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return set()
    return {
        term
        for name in properties
        for term in _tool_terms(str(name).replace("_", " "))
    }


def response_body(
    *,
    response_id: str,
    model: str,
    text: str,
    tool_calls: Sequence[dict[str, Any]],
    usage: dict[str, int],
    metadata: dict[str, Any],
    status: str = "completed",
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if text:
        output.append(response_message_item(response_id, text, status=status))
    output.extend(response_function_item(call, status=status) for call in tool_calls)
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "error": None,
        "incomplete_details": None,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("total_tokens", 0),
        },
        "machboost": metadata,
    }


def response_message_item(response_id: str, text: str, *, status: str) -> dict[str, Any]:
    return {
        "id": f"msg_{response_id.removeprefix('resp_')}",
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }


def response_function_item(call: dict[str, Any], *, status: str) -> dict[str, Any]:
    function = dict(call.get("function") or {})
    return {
        "id": f"fc_{uuid.uuid4().hex[:24]}",
        "type": "function_call",
        "status": status,
        "call_id": str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
        "name": str(function.get("name") or ""),
        "arguments": str(function.get("arguments") or "{}"),
    }


def anthropic_body(
    *,
    message_id: str,
    model: str,
    text: str,
    thinking: str,
    tool_calls: Sequence[dict[str, Any]],
    usage: dict[str, int],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if thinking:
        content.append({"type": "thinking", "thinking": thinking, "signature": ""})
    if text:
        content.append({"type": "text", "text": text})
    for call in tool_calls:
        function = dict(call.get("function") or {})
        arguments = function.get("arguments") or "{}"
        try:
            parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            parsed_arguments = {"raw": arguments}
        content.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
                "name": str(function.get("name") or ""),
                "input": parsed_arguments,
            }
        )
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
        "machboost": metadata,
    }


def _responses_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Responses message content must be text or a parts list")
    result = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Responses content parts must be objects")
        part_type = str(part.get("type") or "input_text")
        if part_type in {"input_text", "output_text", "text"}:
            result.append({"type": "text", "text": str(part.get("text") or "")})
        elif part_type in {"input_image", "image_url", "image"}:
            result.append(
                {
                    "type": "image_url",
                    "image_url": part.get("image_url") or part.get("url") or part.get("image"),
                }
            )
        else:
            raise ValueError(f"unsupported Responses content part: {part_type}")
    return result


def _anthropic_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content or "")


def _anthropic_image_part(part: dict[str, Any]) -> dict[str, Any]:
    source = part.get("source") or {}
    if not isinstance(source, dict):
        raise ValueError("Anthropic image source must be an object")
    source_type = str(source.get("type") or "")
    if source_type == "base64":
        media_type = str(source.get("media_type") or "image/png")
        data = str(source.get("data") or "")
        try:
            base64.b64decode(data, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Anthropic image contains invalid base64 data") from exc
        url = f"data:{media_type};base64,{data}"
    elif source_type == "url":
        url = str(source.get("url") or "")
    else:
        raise ValueError(f"unsupported Anthropic image source: {source_type}")
    return {"type": "image_url", "image_url": {"url": url}}
