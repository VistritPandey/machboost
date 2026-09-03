from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class ThinkingDelta:
    reasoning: str = ""
    content: str = ""


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
