from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

from machboost.core import Token, TokenSeq


@dataclass(frozen=True)
class Verification:
    accepted: int
    residual_token: Optional[Token]


class HuggingFaceCausalLMService:
    def __init__(
        self,
        model,
        tokenizer=None,
        *,
        device: Optional[str] = None,
        min_verify_margin: float = 0.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or _infer_device(model)
        self.min_verify_margin = float(min_verify_margin)
        self.forward_calls = 0
        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        local_files_only: bool = False,
        torch_dtype=None,
        model_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        min_verify_margin: float = 0.0,
    ) -> "HuggingFaceCausalLMService":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install Hugging Face support with `pip install machboost[hf]`.") from exc

        resolved_device = _resolve_device(device)
        model_args = dict(model_kwargs or {})
        tokenizer_args = dict(tokenizer_kwargs or {})
        model_args.setdefault("local_files_only", local_files_only)
        tokenizer_args.setdefault("local_files_only", local_files_only)
        if torch_dtype is None:
            torch_dtype = _default_dtype(resolved_device)
        if torch_dtype is not None:
            model_args["dtype"] = torch_dtype

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_args)
        if resolved_device is not None and hasattr(model, "to"):
            model = model.to(resolved_device)
        return cls(model, tokenizer, device=resolved_device, min_verify_margin=min_verify_margin)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Tuple[Token, ...]:
        if self.tokenizer is None:
            raise ValueError("encode requires a tokenizer")
        return tuple(int(token) for token in self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, tokens: Iterable[Token], *, skip_special_tokens: bool = True) -> str:
        if self.tokenizer is None:
            raise ValueError("decode requires a tokenizer")
        return self.tokenizer.decode(list(tokens), skip_special_tokens=skip_special_tokens)

    def next_token(self, prefix_tokens: TokenSeq) -> Optional[Token]:
        if len(prefix_tokens) == 0:
            return None
        logits = self._logits(prefix_tokens)
        return int(logits[0, -1].argmax().item())

    def generate_tokens(
        self,
        prompt_tokens: TokenSeq,
        *,
        max_tokens: int,
        stop_tokens: Optional[Iterable[Token]] = None,
        on_tokens=None,
        generation_options: Optional[dict[str, Any]] = None,
    ) -> Tuple[Token, ...]:
        if len(prompt_tokens) == 0 or max_tokens <= 0:
            return ()

        torch = _torch()
        stop_set = {int(token) for token in stop_tokens or ()}
        generated: list[Token] = []
        generation_options = dict(generation_options or {})
        if generation_options.get("seed") is not None:
            torch.manual_seed(int(generation_options["seed"]))
        input_ids = torch.tensor([list(prompt_tokens)], dtype=torch.long, device=self.device)
        past_key_values = None

        with torch.inference_mode():
            for step in range(max_tokens):
                if step == 0:
                    out = self.model(input_ids=input_ids, use_cache=True)
                else:
                    out = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                self.forward_calls += 1
                past_key_values = getattr(out, "past_key_values", None)
                token = _sample_token(
                    out.logits[:, -1, :],
                    history=tuple(prompt_tokens) + tuple(generated),
                    options=generation_options,
                )
                if token in stop_set:
                    break
                generated.append(token)
                if on_tokens is not None:
                    on_tokens((token,))
                input_ids = torch.tensor([[token]], dtype=torch.long, device=self.device)

        return tuple(generated)

    def verify(self, prefix_tokens: TokenSeq, candidate_tokens: TokenSeq) -> Tuple[int, Optional[Token]]:
        result = self.verification(prefix_tokens, candidate_tokens)
        return result.accepted, result.residual_token

    def verification(self, prefix_tokens: TokenSeq, candidate_tokens: TokenSeq) -> Verification:
        if len(prefix_tokens) == 0 or len(candidate_tokens) == 0:
            return Verification(0, None)

        sequence = tuple(int(token) for token in prefix_tokens) + tuple(int(token) for token in candidate_tokens)
        logits = self._logits(sequence)
        start = len(prefix_tokens) - 1
        accepted = 0
        residual: Optional[Token] = None

        for offset, candidate in enumerate(candidate_tokens):
            row = logits[0, start + offset]
            predicted = int(row.argmax().item())
            if predicted != int(candidate) or not self._passes_margin(row, int(candidate)):
                residual = predicted
                break
            accepted += 1

        return Verification(accepted, residual)

    def _logits(self, tokens: Sequence[Token]):
        torch = _torch()
        input_ids = torch.tensor([list(tokens)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids)
        self.forward_calls += 1
        return out.logits

    def _passes_margin(self, logits, token: Token) -> bool:
        if self.min_verify_margin <= 0:
            return True
        torch = _torch()
        values = torch.topk(logits, k=min(2, logits.shape[-1])).values
        if values.shape[-1] < 2:
            return True
        top_token = int(logits.argmax().item())
        if top_token != int(token):
            return False
        margin = float((values[0] - values[1]).item())
        return margin >= self.min_verify_margin


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install Hugging Face support with `pip install machboost[hf]`.") from exc
    return torch


def _sample_token(logits, *, history: Sequence[Token], options: dict[str, Any]) -> Token:
    torch = _torch()
    scores = logits[0].clone()
    repeat_last_n = int(options.get("repeat_last_n", 64))
    recent = tuple(history if repeat_last_n < 0 else history[-repeat_last_n:])
    counts: dict[int, int] = {}
    for token in recent:
        counts[int(token)] = counts.get(int(token), 0) + 1
    repeat_penalty = float(options.get("repeat_penalty", 1.0))
    presence_penalty = float(options.get("presence_penalty", 0.0))
    frequency_penalty = float(options.get("frequency_penalty", 0.0))
    for token, count in counts.items():
        if repeat_penalty not in {0.0, 1.0}:
            scores[token] = (
                scores[token] * repeat_penalty
                if scores[token] < 0
                else scores[token] / repeat_penalty
            )
        scores[token] -= presence_penalty + frequency_penalty * count
    temperature = float(options.get("temperature", 0.0))
    if temperature <= 0:
        return int(scores.argmax().item())
    scores = scores / temperature
    top_k = int(options.get("top_k", 0))
    if top_k > 0 and top_k < scores.shape[-1]:
        threshold = torch.topk(scores, top_k).values[-1]
        scores[scores < threshold] = -float("inf")
    probabilities = torch.softmax(scores, dim=-1)
    top_p = float(options.get("top_p", 0.0))
    if 0.0 < top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities > top_p
        sorted_probabilities[remove] = 0
        probabilities = torch.zeros_like(probabilities).scatter(
            0, sorted_indices, sorted_probabilities
        )
    min_p = float(options.get("min_p", 0.0))
    if min_p > 0:
        probabilities[probabilities < probabilities.max() * min_p] = 0
    total = probabilities.sum()
    if not bool(torch.isfinite(total)) or float(total.item()) <= 0:
        return int(scores.argmax().item())
    return int(torch.multinomial(probabilities / total, 1).item())


def _resolve_device(device: Optional[str]) -> Optional[str]:
    if device and device != "auto":
        return device
    torch = _torch()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return None


def _default_dtype(device: Optional[str]):
    if device == "mps":
        return _torch().float16
    return None


def _infer_device(model) -> str:
    try:
        param = next(model.parameters())
    except (AttributeError, StopIteration):
        return "cpu"
    return str(param.device)
