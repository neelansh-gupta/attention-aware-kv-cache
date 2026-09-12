"""Stage 9 quality evaluation: perplexity and needle-in-a-haystack.

All policies share the same teacher-forced, token-by-token cache walk.
Compressed policies evict after every token. The full-cache policy never
evicts. Perplexity is exp(mean next-token NLL). NIH success is whether the
unique passcode appears in the greedy continuation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .cache_manager import (
    AttentionSinkCacheManager,
    HeavyHitterCacheManager,
    SlidingWindowCacheManager,
)
from .model_wrapper import (
    ModelWrapper,
    aggregate_newest_attention,
    update_accumulated_scores,
)
from .position_utils import build_absolute_position_ids, build_cache_position


@dataclass
class StreamState:
    past: Any
    positions: torch.Tensor
    scores: torch.Tensor | None
    next_position: int
    attention_mask: torch.Tensor
    manager: Any
    policy: str
    cache_budget: int


def build_manager(
    policy: str,
    cache_budget: int,
    sink_tokens: int,
    h2o_sink_tokens: int,
    h2o_recent_window: int,
):
    if policy == "full":
        return None
    if policy == "sliding":
        return SlidingWindowCacheManager(window_size=cache_budget)
    if policy == "streaming":
        return AttentionSinkCacheManager(
            cache_budget=cache_budget,
            sink_tokens=sink_tokens,
        )
    if policy == "h2o":
        return HeavyHitterCacheManager(
            cache_budget=cache_budget,
            sink_tokens=h2o_sink_tokens,
            recent_window=h2o_recent_window,
        )
    raise ValueError(f"Unknown policy: {policy}")


def new_state(
    wrapper: ModelWrapper,
    policy: str,
    cache_budget: int,
    sink_tokens: int,
    h2o_sink_tokens: int,
    h2o_recent_window: int,
) -> StreamState:
    return StreamState(
        past=None,
        positions=torch.empty(0, dtype=torch.long),
        scores=None,
        next_position=0,
        attention_mask=torch.ones((1, 0), dtype=torch.long, device=wrapper.device),
        manager=build_manager(
            policy,
            cache_budget,
            sink_tokens,
            h2o_sink_tokens,
            h2o_recent_window,
        ),
        policy=policy,
        cache_budget=cache_budget,
    )


@torch.no_grad()
def step_token(
    wrapper: ModelWrapper,
    state: StreamState,
    token: torch.Tensor,
) -> torch.Tensor:
    """Consume one token, evict if needed, return logits for the next token."""

    current = token.view(1, 1).to(wrapper.device)
    position_ids = build_absolute_position_ids(
        start_position=state.next_position,
        sequence_length=1,
        device=wrapper.device,
    )
    cache_position = build_cache_position(
        start_position=state.next_position,
        sequence_length=1,
        device=wrapper.device,
    )
    state.attention_mask = torch.cat(
        [
            state.attention_mask,
            torch.ones((1, 1), dtype=torch.long, device=wrapper.device),
        ],
        dim=-1,
    )
    need_attention = state.policy == "h2o"
    kwargs = {
        "input_ids": current,
        "attention_mask": state.attention_mask,
        "position_ids": position_ids,
        "past_key_values": state.past,
        "use_cache": True,
        "output_attentions": need_attention,
    }
    try:
        outputs = wrapper.model(**kwargs, cache_position=cache_position)
    except (TypeError, ValueError):
        outputs = wrapper.model(**kwargs)

    state.next_position += 1
    state.past = outputs.past_key_values
    step_positions = torch.cat((state.positions, position_ids[0].detach().cpu()))

    if state.policy == "h2o":
        current_attention = aggregate_newest_attention(outputs.attentions)
        state.scores = update_accumulated_scores(state.scores, current_attention)
        state.past, state.scores = state.manager.update(
            state.past,
            state.scores,
            token_positions=step_positions,
        )
        state.positions = state.manager.retained_positions
        cache_length = state.manager.sequence_length(state.past)
    elif state.manager is None:
        state.positions = step_positions
        cache_length = int(state.past.get_seq_length())
    else:
        state.past = state.manager.update(
            state.past,
            token_positions=step_positions,
        )
        state.positions = state.manager.retained_positions
        cache_length = state.manager.sequence_length(state.past)

    expected_mask_length = cache_length + 1
    if state.attention_mask.shape[-1] > expected_mask_length:
        state.attention_mask = state.attention_mask[:, -expected_mask_length:]

    return outputs.logits[:, -1, :].float()


def prefill(
    wrapper: ModelWrapper,
    state: StreamState,
    token_ids: torch.Tensor,
) -> torch.Tensor | None:
    logits = None
    for token in token_ids.view(-1):
        logits = step_token(wrapper, state, token)
    return logits


def teacher_forced_perplexity(
    wrapper: ModelWrapper,
    token_ids: torch.Tensor,
    policy: str,
    cache_budget: int,
    sink_tokens: int,
    h2o_sink_tokens: int,
    h2o_recent_window: int,
) -> dict[str, Any]:
    if token_ids.numel() < 2:
        raise ValueError("Perplexity needs at least two tokens.")
    state = new_state(
        wrapper,
        policy,
        cache_budget,
        sink_tokens,
        h2o_sink_tokens,
        h2o_recent_window,
    )
    nlls: list[float] = []
    tokens = token_ids.view(-1)
    for index in range(tokens.numel() - 1):
        logits = step_token(wrapper, state, tokens[index])
        target = tokens[index + 1].view(1).to(wrapper.device)
        nlls.append(float(F.cross_entropy(logits, target).item()))
    mean_nll = sum(nlls) / len(nlls)
    return {
        "policy": policy,
        "cache_budget": None if policy == "full" else cache_budget,
        "tokens": int(tokens.numel()),
        "nll_tokens": len(nlls),
        "mean_nll": mean_nll,
        "perplexity": float(torch.exp(torch.tensor(mean_nll)).item()),
        "final_cache_length": int(state.positions.numel()),
        "retained_positions": state.positions.tolist(),
    }


def greedy_continue(
    wrapper: ModelWrapper,
    state: StreamState,
    first_logits: torch.Tensor,
    max_new_tokens: int,
) -> list[int]:
    generated: list[int] = []
    logits = first_logits
    eos = wrapper.tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        token_id = int(torch.argmax(logits, dim=-1).item())
        generated.append(token_id)
        if eos is not None and token_id == eos:
            break
        logits = step_token(
            wrapper,
            state,
            torch.tensor([token_id], device=wrapper.device),
        )
    return generated


def needle_success(generated_text: str, passcode: str) -> bool:
    compact = "".join(ch for ch in generated_text.upper() if ch.isalnum())
    needle = "".join(ch for ch in passcode.upper() if ch.isalnum())
    return needle in compact
