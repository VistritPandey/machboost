from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

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

        model_args = dict(model_kwargs or {})
        tokenizer_args = dict(tokenizer_kwargs or {})
        model_args.setdefault("local_files_only", local_files_only)
        tokenizer_args.setdefault("local_files_only", local_files_only)
        if torch_dtype is not None:
            model_args["torch_dtype"] = torch_dtype

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_args)
        if device is not None and hasattr(model, "to"):
            model = model.to(device)
        return cls(model, tokenizer, device=device, min_verify_margin=min_verify_margin)

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


def _infer_device(model) -> str:
    try:
        param = next(model.parameters())
    except (AttributeError, StopIteration):
        return "cpu"
    return str(param.device)
