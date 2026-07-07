import argparse

import torch

from machboost import machboost
from machboost.adapters import HuggingFaceCausalLMService


class Output:
    def __init__(self, logits):
        self.logits = logits


class TinyCausalModel(torch.nn.Module):
    def __init__(self, target_tokens, prompt_len):
        super().__init__()
        self.target_tokens = tuple(target_tokens)
        self.prompt_len = prompt_len
        self.vocab_size = max(self.target_tokens + (0,)) + 8
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids):
        batch, seqlen = input_ids.shape
        logits = torch.zeros(batch, seqlen, self.vocab_size, device=input_ids.device)
        for pos in range(seqlen):
            target_offset = pos - self.prompt_len + 1
            token = 0
            if 0 <= target_offset < len(self.target_tokens):
                token = self.target_tokens[target_offset]
            logits[:, pos, token] = 10.0
        return Output(logits)


def run_toy(args):
    prompt = (100, 101, 102)
    target = (1, 2, 3, 4, 5, 6, 7, 8)
    service = HuggingFaceCausalLMService(TinyCausalModel(target, prompt_len=len(prompt)))
    boosted = machboost(
        service,
        corpus_tokens=prompt + target,
        ngram=args.ngram,
        max_draft_tokens=args.max_draft_tokens,
    )

    generated, stats = boosted.generate(prompt, max_tokens=len(target))
    print("mode: toy-hf-adapter")
    print("generated:", generated)
    print("exact:", generated == target)
    print("target forwards:", service.forward_calls)
    print("estimated speedup:", f"{stats.estimated_speedup:.2f}x")


def run_real_model(args):
    service = HuggingFaceCausalLMService.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    prompt_tokens = service.encode(args.prompt)
    context_text = args.context or args.prompt
    corpus_tokens = prompt_tokens + service.encode(context_text)
    boosted = machboost(
        service,
        corpus_tokens=corpus_tokens,
        ngram=args.ngram,
        max_draft_tokens=args.max_draft_tokens,
    )

    generated, stats = boosted.generate(prompt_tokens, max_tokens=args.max_tokens)
    print("mode: transformers")
    print("generated text:")
    print(service.decode(generated))
    print("target forwards:", service.forward_calls)
    print("accepted draft tokens:", stats.accepted_draft_tokens)
    print("estimated speedup:", f"{stats.estimated_speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Run MachBoost with the Hugging Face verifier adapter.")
    parser.add_argument("--model", help="Hugging Face model name/path. Omit to run a deterministic toy model.")
    parser.add_argument("--prompt", default="Write the repeated sequence:", help="Prompt for real-model mode.")
    parser.add_argument("--context", default="", help="Local context text for real-model drafting.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ngram", type=int, default=4)
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.model:
        run_real_model(args)
    else:
        run_toy(args)


if __name__ == "__main__":
    main()
