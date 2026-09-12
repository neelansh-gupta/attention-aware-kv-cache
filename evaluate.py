"""Stage 9 CLI: perplexity and needle-in-a-haystack under a fixed cache budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.evaluation import (
    greedy_continue,
    needle_success,
    new_state,
    prefill,
    teacher_forced_perplexity,
)
from src.model_wrapper import ModelWrapper


PASSCODE = "ZX9QWERTY7731"
NEEDLE = (
    f" Remember this unique fact: the secret passcode is {PASSCODE}."
)
QUESTION = " Question: What is the secret passcode? Answer:"
FILLER = (
    " The archive described ordinary weather, rivers, markets, and local "
    "history without mentioning any secret codes or hidden instructions."
)
LONG_DOCUMENT = (
    "Language models store keys and values for previous tokens so later "
    "predictions can attend to earlier context. When that cache is compressed, "
    "some tokens are dropped. Measuring next-token loss on a long document "
    "shows whether remaining entries still support fluent local prediction. "
) * 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 9 quality evaluation.")
    parser.add_argument("--task", choices=("perplexity", "nih", "all"), default="all")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--nih-generate-tokens", type=int, default=24)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--h2o-sink-tokens", type=int, default=1)
    parser.add_argument("--h2o-recent-window", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    return parser.parse_args()


def encode_ids(wrapper: ModelWrapper, text: str) -> torch.Tensor:
    return wrapper.tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].to(wrapper.device)


def clip_tokens(wrapper: ModelWrapper, text: str, max_tokens: int) -> torch.Tensor:
    token_ids = encode_ids(wrapper, text)[:max_tokens]
    if token_ids.numel() < 2:
        raise ValueError("Tokenized text is too short.")
    return token_ids


def build_haystack(
    wrapper: ModelWrapper,
    depth: str,
    body_tokens: int,
) -> dict:
    filler_ids = encode_ids(wrapper, FILLER * 40)
    needle_ids = encode_ids(wrapper, NEEDLE)
    question_ids = encode_ids(wrapper, QUESTION)
    if filler_ids.numel() < body_tokens:
        raise ValueError("Not enough filler tokens for the requested haystack.")
    body = filler_ids[:body_tokens]
    needle_len = int(needle_ids.numel())
    if depth == "start":
        insert_at = 0
    elif depth == "middle":
        insert_at = max(0, body.numel() // 2 - needle_len // 2)
    elif depth == "end":
        insert_at = max(0, body.numel() - needle_len)
    else:
        raise ValueError(f"Unknown depth: {depth}")
    context = torch.cat((body[:insert_at], needle_ids, body[insert_at:], question_ids))
    needle_positions = list(range(insert_at, insert_at + needle_len))
    return {
        "depth": depth,
        "token_ids": context,
        "needle_positions": needle_positions,
        "prompt_tokens": int(context.numel()),
        "needle_token_count": needle_len,
    }


def run_perplexity(wrapper: ModelWrapper, args: argparse.Namespace) -> dict:
    token_ids = clip_tokens(wrapper, LONG_DOCUMENT, args.max_tokens)
    policies = ("full", "sliding", "streaming", "h2o")
    rows = []
    for policy in policies:
        print(f"Perplexity policy={policy}", flush=True)
        row = teacher_forced_perplexity(
            wrapper,
            token_ids,
            policy,
            args.budget,
            args.sink_tokens,
            args.h2o_sink_tokens,
            args.h2o_recent_window,
        )
        rows.append(row)
        print(
            f"  ppl={row['perplexity']:.4f} mean_nll={row['mean_nll']:.4f} "
            f"cache_length={row['final_cache_length']}",
            flush=True,
        )
    return {
        "methodology": (
            "Teacher-forced next-token NLL while walking the same document "
            "token by token. Compressed policies evict after every token. "
            "full never evicts. perplexity = exp(mean NLL)."
        ),
        "document_tokens": int(token_ids.numel()),
        "cache_budget": args.budget,
        "results": rows,
    }


def run_nih(wrapper: ModelWrapper, args: argparse.Namespace) -> dict:
    depths = ("start", "middle", "end")
    policies = ("full", "sliding", "streaming", "h2o")
    haystack_body = max(args.budget + 16, 48)
    rows = []
    for depth in depths:
        haystack = build_haystack(wrapper, depth, haystack_body)
        for policy in policies:
            print(f"NIH depth={depth} policy={policy}", flush=True)
            state = new_state(
                wrapper,
                policy,
                args.budget,
                args.sink_tokens,
                args.h2o_sink_tokens,
                args.h2o_recent_window,
            )
            logits = prefill(wrapper, state, haystack["token_ids"])
            generated_ids = greedy_continue(
                wrapper,
                state,
                logits,
                args.nih_generate_tokens,
            )
            text = wrapper.tokenizer.decode(generated_ids, skip_special_tokens=True)
            retained = set(state.positions.tolist())
            needle_retained = [
                position
                for position in haystack["needle_positions"]
                if position in retained
            ]
            success = needle_success(text, PASSCODE)
            row = {
                "depth": depth,
                "policy": policy,
                "cache_budget": None if policy == "full" else args.budget,
                "prompt_tokens": haystack["prompt_tokens"],
                "needle_positions": haystack["needle_positions"],
                "needle_tokens_retained": needle_retained,
                "needle_retained_fraction": (
                    len(needle_retained) / max(1, len(haystack["needle_positions"]))
                ),
                "generated_token_ids": generated_ids,
                "generated_text": text,
                "success": success,
                "final_cache_length": int(state.positions.numel()),
            }
            rows.append(row)
            print(
                f"  success={success} retained_needle="
                f"{len(needle_retained)}/{len(haystack['needle_positions'])} "
                f"text={text!r}",
                flush=True,
            )
    return {
        "methodology": (
            "Insert a unique passcode at start/middle/end of a filler haystack, "
            "then ask for the passcode. Success is whether the passcode string "
            "appears in the greedy continuation after cache-compressed prefill."
        ),
        "passcode": PASSCODE,
        "cache_budget": args.budget,
        "haystack_body_tokens": haystack_body,
        "generate_tokens": args.nih_generate_tokens,
        "results": rows,
    }


def main() -> None:
    args = parse_args()
    if args.budget <= 0:
        raise ValueError("--budget must be positive.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = ModelWrapper(
        model_name=args.model or "Qwen/Qwen2.5-0.5B",
        device=args.device,
    )
    payload = {
        "stage": 9,
        "model": wrapper.model_name,
        "device": str(wrapper.device),
        "sink_tokens": args.sink_tokens,
        "h2o_sink_tokens": args.h2o_sink_tokens,
        "h2o_recent_window": args.h2o_recent_window,
    }
    perplexity_path = output_dir / "stage9_perplexity.json"
    nih_path = output_dir / "stage9_nih.json"
    if args.task in {"perplexity", "all"}:
        payload["perplexity"] = run_perplexity(wrapper, args)
        perplexity_path.write_text(json.dumps(payload["perplexity"], indent=2) + "\n")
    elif perplexity_path.exists():
        payload["perplexity"] = json.loads(perplexity_path.read_text())
    if args.task in {"nih", "all"}:
        payload["needle_in_a_haystack"] = run_nih(wrapper, args)
        nih_path.write_text(json.dumps(payload["needle_in_a_haystack"], indent=2) + "\n")
    elif nih_path.exists():
        payload["needle_in_a_haystack"] = json.loads(nih_path.read_text())
    (output_dir / "stage9_raw.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {output_dir / 'stage9_raw.json'}")


if __name__ == "__main__":
    main()
