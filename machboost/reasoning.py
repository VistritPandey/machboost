from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ThinkingDelta:
    reasoning: str = ""
    content: str = ""


@dataclass(frozen=True)
class ThinkingProtocol:
    start_marker: Optional[str] = None
    end_marker: Optional[str] = None

    def starts_in_thinking(self, enabled: bool) -> bool:
        return bool(enabled) and self.start_marker in {
            None,
            "<think>",
            "<|START_THINKING|>",
        }


def resolve_thinking_protocol(config: Any, tokenizer: Any) -> ThinkingProtocol:
    """Resolve reasoning delimiters from a model's config or response schema."""
    start_marker = _value(config, "thinking_start_token")
    end_marker = _value(config, "thinking_end_token")
    response_template = _value(tokenizer, "response_template")
    if not isinstance(response_template, Mapping):
        init_kwargs = _value(tokenizer, "init_kwargs")
        if isinstance(init_kwargs, Mapping):
            response_template = init_kwargs.get("response_template")
    if isinstance(response_template, Mapping):
        fields = response_template.get("fields")
        reasoning = fields.get("reasoning_content") if isinstance(fields, Mapping) else None
        if isinstance(reasoning, Mapping):
            schema_start = _literal_response_marker(reasoning.get("open_pattern"))
            schema_end = reasoning.get("close")
            if schema_start:
                start_marker = schema_start
            if isinstance(schema_end, str) and schema_end:
                end_marker = schema_end
            elif isinstance(schema_end, Sequence) and not isinstance(schema_end, str):
                end_marker = next(
                    (value for value in schema_end if isinstance(value, str) and value),
                    end_marker,
                )

    model_type = str(_value(config, "model_type") or "").lower()
    if model_type == "muse_glimmer":
        start_marker = start_marker or "to=self<|message|>"
        end_marker = end_marker or "<|eom|>"
    return ThinkingProtocol(start_marker=start_marker, end_marker=end_marker)


def _value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _literal_response_marker(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    marker = value.replace(r"\|", "|")
    if any(character in marker for character in "[](){}?+*^$") or r"\b" in marker:
        return None
    return marker


class ThinkingStreamSplitter:
    """Separate model reasoning markers without leaking partial tags."""

    DEFAULT_MARKERS = (
        ("<|channel>thought", "<channel|>"),
        ("<think>", "</think>"),
        ("<|START_THINKING|>", "<|END_THINKING|>"),
    )
    CONTENT_MARKERS = (
        "<|start|>assistant to=user<|message|>",
        "<|start|>assistant<|message|>",
        "assistant to=user<|message|>",
        "to=user<|message|>",
        "<|START_TEXT|>",
        "<|END_TEXT|>",
        "<|eot|>",
    )

    def __init__(
        self,
        *,
        starts_in_thinking: bool = False,
        start_marker: Optional[str] = None,
        end_marker: Optional[str] = None,
    ) -> None:
        markers: list[tuple[str, str]] = []
        if start_marker and end_marker:
            if start_marker.startswith("to="):
                markers.append((f"<|start|>assistant {start_marker}", end_marker))
            markers.append((start_marker, end_marker))
        markers.extend(pair for pair in self.DEFAULT_MARKERS if pair not in markers)
        self.markers = tuple(markers)
        self.open_markers = tuple(pair[0] for pair in self.markers)
        self.close_markers = tuple(pair[1] for pair in self.markers)
        self.in_thinking = starts_in_thinking
        self.thinking_done = False
        self.buffer = ""

    def feed(self, text: str, *, final: bool = False) -> ThinkingDelta:
        self.buffer += text
        reasoning: list[str] = []
        content: list[str] = []
        while self.buffer:
            if self.in_thinking:
                index, marker = self._find_first(self.buffer, self.close_markers)
                if index < 0:
                    value, self.buffer = self._split_partial(
                        self.buffer, self.close_markers, final=final
                    )
                    if value:
                        reasoning.append(self._strip_open_marker(value))
                    break
                value = self._strip_open_marker(self.buffer[:index])
                if value:
                    reasoning.append(value)
                self.buffer = self.buffer[index + len(marker) :].lstrip("\n")
                self.in_thinking = False
                self.thinking_done = True
                continue

            if self.thinking_done:
                value, self.buffer = self._split_partial(
                    self.buffer, self.CONTENT_MARKERS, final=final
                )
                if value:
                    content.append(self._clean_content(value))
                break

            index, marker = self._find_first(self.buffer, self.open_markers)
            if index < 0:
                value, self.buffer = self._split_partial(
                    self.buffer,
                    self.open_markers + self.CONTENT_MARKERS,
                    final=final,
                )
                if value:
                    content.append(self._clean_content(value))
                break
            if index:
                content.append(self._clean_content(self.buffer[:index]))
            self.buffer = self.buffer[index + len(marker) :].lstrip("\n")
            self.in_thinking = True

        return ThinkingDelta("".join(reasoning), "".join(content))

    @staticmethod
    def _find_first(text: str, markers: Sequence[str]) -> tuple[int, str]:
        matches = ((text.find(marker), marker) for marker in markers)
        return min(
            ((index, marker) for index, marker in matches if index >= 0),
            default=(-1, ""),
            key=lambda match: match[0],
        )

    @staticmethod
    def _split_partial(
        text: str, markers: Sequence[str], *, final: bool
    ) -> tuple[str, str]:
        if final or not markers:
            return text, ""
        hold = 0
        for marker in markers:
            for length in range(1, min(len(text), len(marker) - 1) + 1):
                if text.endswith(marker[:length]):
                    hold = max(hold, length)
        if hold:
            return text[:-hold], text[-hold:]
        return text, ""

    def _strip_open_marker(self, text: str) -> str:
        for marker in self.open_markers:
            text = text.replace(marker, "")
        return text

    def _clean_content(self, text: str) -> str:
        for marker in self.CONTENT_MARKERS:
            text = text.replace(marker, "")
        return text
