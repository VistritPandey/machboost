from __future__ import annotations

import json
import math
from typing import Any, Callable, Optional, Sequence


OLLAMA_OPTION_KEYS = {
    "num_ctx",
    "num_predict",
    "num_keep",
    "seed",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repeat_last_n",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "truncate",
    "shift",
}


def normalize_ollama_options(payload: dict[str, Any]) -> dict[str, Any]:
    options = {
        key: value
        for key, value in dict(payload.get("options") or {}).items()
        if value is not None
    }
    for key in OLLAMA_OPTION_KEYS:
        if key in payload and payload[key] is not None and key not in options:
            options[key] = payload[key]
    if "format" in payload:
        options["_format"] = normalize_format(payload["format"])
    if "think" in payload:
        think = payload["think"]
        if not isinstance(think, (bool, str)):
            raise ValueError("think must be a boolean or level string")
        options["_think"] = think
        if isinstance(think, str):
            options["_reasoning_strength"] = think
    if "raw" in payload:
        options["_raw"] = bool(payload["raw"])
    if "system" in payload:
        options["_system"] = str(payload["system"])
    if "template" in payload:
        options["_template"] = str(payload["template"])
    _validate_options(options)
    return options


def truncate_prompt(
    service: Any,
    prompt: str,
    *,
    num_ctx: Optional[int],
    max_tokens: int,
    truncate: bool = True,
    num_keep: int = 0,
) -> tuple[str, int]:
    if num_ctx is None:
        return prompt, 0
    budget = int(num_ctx) - max(0, int(max_tokens))
    if budget < 1:
        raise ValueError("num_ctx must leave room for at least one prompt token")
    tokens = tuple(service.encode(prompt))
    if len(tokens) <= budget:
        return prompt, 0
    if not truncate:
        raise ValueError(
            f"input requires {len(tokens)} tokens but num_ctx leaves {budget} for the prompt"
        )
    keep = min(max(0, int(num_keep)), budget, len(tokens))
    tail_count = budget - keep
    selected = tokens[:keep] + (tokens[-tail_count:] if tail_count else ())
    return str(service.decode(selected, skip_special_tokens=False)), len(tokens) - len(selected)


def truncate_messages(
    service: Any,
    messages: Sequence[dict[str, Any]],
    *,
    render: Callable[[Sequence[dict[str, Any]]], str],
    num_ctx: Optional[int],
    max_tokens: int,
    truncate: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    result = [dict(message) for message in messages]
    if num_ctx is None:
        return result, 0
    budget = int(num_ctx) - max(0, int(max_tokens))
    if budget < 1:
        raise ValueError("num_ctx must leave room for at least one prompt token")

    def token_count() -> int:
        return len(service.encode(render(result)))

    initial = token_count()
    if initial <= budget:
        return result, 0
    if not truncate:
        raise ValueError(
            f"chat requires {initial} tokens but num_ctx leaves {budget} for the prompt"
        )
    while token_count() > budget:
        removable = next(
            (
                index
                for index, message in enumerate(result[:-1])
                if str(message.get("role") or "") != "system"
            ),
            None,
        )
        if removable is None:
            break
        result.pop(removable)
    current = token_count()
    if current > budget and result:
        target = next(
            (
                message
                for message in reversed(result)
                if str(message.get("role") or "") != "system"
                and isinstance(message.get("content"), str)
            ),
            None,
        )
        if target is None:
            raise ValueError(
                "system messages and chat template exceed num_ctx; increase num_ctx or shorten them"
            )
        content_tokens = tuple(service.encode(str(target["content"])))
        overflow = current - budget
        if overflow >= len(content_tokens):
            target["content"] = ""
        else:
            target["content"] = service.decode(
                content_tokens[overflow:], skip_special_tokens=False
            )
    final = token_count()
    if final > budget:
        raise ValueError(
            "system messages and chat template exceed num_ctx; increase num_ctx or shorten them"
        )
    return result, max(0, initial - final)


def apply_generate_template(prompt: str, options: dict[str, Any]) -> str:
    system = str(options.get("_system") or "")
    template = str(options.get("_template") or "")
    if template:
        return (
            template.replace("{{ .System }}", system)
            .replace("{{.System}}", system)
            .replace("{{ .Prompt }}", prompt)
            .replace("{{.Prompt}}", prompt)
            .replace("{{ .Response }}", "")
            .replace("{{.Response}}", "")
        )
    if system:
        return f"System: {system}\n\nUser: {prompt}\n\nAssistant:"
    return prompt


def structured_output_instruction(format_value: Any) -> str:
    normalized = normalize_format(format_value)
    if normalized is None:
        return ""
    if normalized == "json":
        return "Return only valid JSON. Do not wrap it in Markdown fences."
    return (
        "Return only JSON matching this JSON Schema. Do not wrap it in Markdown fences.\n"
        + json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    )


def validate_structured_output(text: str, format_value: Any) -> Any:
    normalized = normalize_format(format_value)
    if normalized is None:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model returned invalid JSON: {exc.msg}") from exc
    if isinstance(normalized, dict):
        _validate_schema_value(value, normalized, path="$")
    return value


def normalize_format(value: Any) -> Any:
    if value in (None, ""):
        return None
    if value == "json":
        return "json"
    if isinstance(value, dict):
        return json.loads(json.dumps(value))
    raise ValueError("format must be 'json' or a JSON Schema object")


def _validate_options(options: dict[str, Any]) -> None:
    integer_minimums = {
        "num_ctx": 1,
        "num_predict": -2,
        "num_keep": 0,
        "top_k": 0,
        "repeat_last_n": -1,
    }
    for key, minimum in integer_minimums.items():
        if key not in options:
            continue
        value = int(options[key])
        if value < minimum:
            raise ValueError(f"{key} must be at least {minimum}")
        options[key] = value
    bounded = {"top_p": (0.0, 1.0), "min_p": (0.0, 1.0)}
    for key, (minimum, maximum) in bounded.items():
        if key not in options:
            continue
        value = float(options[key])
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        options[key] = value
    for key in (
        "temperature",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    ):
        if key not in options:
            continue
        value = float(options[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if key in {"temperature", "repeat_penalty"} and value < 0:
            raise ValueError(f"{key} cannot be negative")
        options[key] = value
    if "stop" in options:
        stop = options["stop"]
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, (list, tuple)) or not all(
            isinstance(item, str) for item in stop
        ):
            raise ValueError("stop must be a string or list of strings")
        options["stop"] = list(dict.fromkeys(stop))


def _validate_schema_value(value: Any, schema: dict[str, Any], *, path: str) -> None:
    expected = schema.get("type")
    checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected in checks and not checks[expected](value):
        raise ValueError(f"model JSON does not match schema at {path}: expected {expected}")
    if isinstance(value, dict):
        required = schema.get("required") or ()
        for key in required:
            if key not in value:
                raise ValueError(f"model JSON does not match schema at {path}: missing {key}")
        properties = schema.get("properties") or {}
        for key, child in properties.items():
            if key in value and isinstance(child, dict):
                _validate_schema_value(value[key], child, path=f"{path}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate_schema_value(item, schema["items"], path=f"{path}[{index}]")
