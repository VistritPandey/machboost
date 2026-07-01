from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from machboost.core import Token, TokenSeq


@dataclass(frozen=True)
class Verification:
    accepted: int
    residual_token: Optional[Token]


class MLXCausalLMService:
    def __init__(
        self,
        model,
        tokenizer=None,
        *,
        mx_module=None,
        min_verify_margin: float = 0.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.mx = mx_module
        self.min_verify_margin = float(min_verify_margin)
        self.forward_calls = 0
        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        tokenizer_config: Optional[dict] = None,
        model_config: Optional[dict] = None,
        adapter_path: Optional[str] = None,
        lazy: bool = False,
        revision: Optional[str] = None,
        min_verify_margin: float = 0.0,
    ) -> "MLXCausalLMService":
        try:
            from mlx_lm.utils import load
        except ImportError as exc:
            raise ImportError("Install MLX support with `pip install machboost[mlx]`.") from exc

        model, tokenizer = load(
            model_name_or_path,
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            adapter_path=adapter_path,
            lazy=lazy,
            revision=revision,
        )
        return cls(model, tokenizer, min_verify_margin=min_verify_margin)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Tuple[Token, ...]:
        if self.tokenizer is None:
            raise ValueError("encode requires a tokenizer")
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:
            tokens = self.tokenizer.encode(text)
        return tuple(int(token) for token in tokens)

    def decode(self, tokens: Iterable[Token], *, skip_special_tokens: bool = True) -> str:
        if self.tokenizer is None:
            raise ValueError("decode requires a tokenizer")
        try:
            return self.tokenizer.decode(list(tokens), skip_special_tokens=skip_special_tokens)
        except TypeError:
            return self.tokenizer.decode(list(tokens))

    def next_token(self, prefix_tokens: TokenSeq) -> Optional[Token]:
        if len(prefix_tokens) == 0:
            return None
        logits = self._logits(prefix_tokens)
        return self._argmax(self._row(logits, len(prefix_tokens) - 1))

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
            row = self._row(logits, start + offset)
            predicted = self._argmax(row)
            if predicted != int(candidate) or not self._passes_margin(row, int(candidate)):
                residual = predicted
                break
            accepted += 1

        return Verification(accepted, residual)

    def _logits(self, tokens: Sequence[Token]):
        mx = self._mx()
        input_ids = self._array([[int(token) for token in tokens]], mx)
        logits = self.model(input_ids)
        self.forward_calls += 1
        if hasattr(mx, "eval"):
            mx.eval(logits)
        return logits

    def _mx(self):
        if self.mx is not None:
            return self.mx
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise ImportError("Install MLX support with `pip install machboost[mlx]`.") from exc
        self.mx = mx
        return mx

    def _array(self, values, mx):
        dtype = getattr(mx, "int32", None)
        if dtype is None:
            return mx.array(values)
        return mx.array(values, dtype=dtype)

    def _row(self, logits, pos: int):
        try:
            return logits[0, pos]
        except (TypeError, IndexError):
            return logits[0][pos]

    def _argmax(self, row) -> Token:
        mx = self._mx()
        if hasattr(mx, "argmax"):
            value = mx.argmax(row)
            if hasattr(value, "item"):
                return int(value.item())
            return int(value)
        return max(range(len(row)), key=lambda i: row[i])

    def _passes_margin(self, row, token: Token) -> bool:
        if self.min_verify_margin <= 0:
            return True
        predicted = self._argmax(row)
        if predicted != int(token):
            return False
        top = sorted((float(v) for v in row), reverse=True)[:2]
        if len(top) < 2:
            return True
        return top[0] - top[1] >= self.min_verify_margin
