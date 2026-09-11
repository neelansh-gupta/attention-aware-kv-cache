from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache_utils import cache_as_tensor_tuple
from .cache_manager import (
    AttentionSinkCacheManager,
    HeavyHitterCacheManager,
    SlidingWindowCacheManager,
)

from .position_utils import (
    build_absolute_position_ids,
    build_cache_position,
)


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@dataclass
class CacheInfo:
    """
    Basic information about the model KV cache.
    """

    num_layers: int
    sequence_length: int
    cache_type: str
    key_shape: tuple
    value_shape: tuple


@dataclass
class AttentionAnalysis:
    """
    Container for attention-analysis results.
    """

    attention_received: torch.Tensor
    early_token_attention: torch.Tensor
    most_attended_token: int
    attention_matrix: torch.Tensor
    attention_by_layer: tuple[torch.Tensor, ...]
    attention_by_head: torch.Tensor


@dataclass
class GenerationResult:
    """Generation text plus cache diagnostics used by correctness tests."""

    text: str
    generated_token_ids: list[int]
    cache_length: int
    retained_positions: list[int]
    score_updates: int = 0


def aggregate_newest_attention(attentions: Any) -> torch.Tensor:
    """Average the newest query's attention across layers and heads."""

    if attentions is None or len(attentions) == 0:
        raise ValueError("No attention tensors were provided.")
    per_layer = []
    for layer_attention in attentions:
        if layer_attention.dim() != 4:
            raise ValueError(
                "Expected attention tensor with shape [batch, heads, query, key]."
            )
        per_layer.append(layer_attention[:, :, -1, :].float().mean(dim=1)[0])
    return torch.stack(per_layer).mean(dim=0)


def update_accumulated_scores(
    accumulated_scores: torch.Tensor | None,
    current_attention: torch.Tensor,
) -> torch.Tensor:
    """Add one decode step while initializing any newly cached token."""

    current_attention = current_attention.flatten()
    if accumulated_scores is None:
        return current_attention.clone()
    previous_length = accumulated_scores.numel()
    if current_attention.numel() == previous_length + 1:
        return torch.cat(
            (
                accumulated_scores + current_attention[:previous_length],
                current_attention[previous_length:],
            )
        )
    if current_attention.numel() == previous_length:
        return accumulated_scores + current_attention
    raise ValueError(
        "Attention sequence length does not match the tracked KV-cache length."
    )


class ModelWrapper:
    """
    Wrapper around a Hugging Face causal language model.

    Stages implemented:
        Stage 1:
            - Model loading
            - Tokenization
            - Incremental generation
            - KV-cache inspection

        Stage 2:
            - Attention extraction
            - Attention aggregation
            - Attention-sink analysis

        Stage 3:
            - Sliding-window KV-cache eviction

        Stage 4:
            - StreamingLLM-style attention-sink-aware KV-cache eviction

        Stage 5:
            - H2O-style accumulated-attention heavy-hitter KV-cache eviction

        Stage 6:
            - Absolute position tracking after KV-cache eviction
            - Correct position_ids for continued generation
            - Preserve already-rotated cached keys at original positions
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
        )

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(
        self,
        text: str,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize input text and move tensors to the model device.
        """

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )

        return {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

    # ------------------------------------------------------------------
    # Stage 1: Incremental generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def incremental_generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        return_diagnostics: bool = False,
    ) -> str | GenerationResult:
        """
        Generate tokens incrementally using the model's KV cache.
        """

        inputs = self.tokenize(prompt)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        generated_ids = input_ids.clone()

        past_key_values = None

        for _ in range(max_new_tokens):

            if past_key_values is None:
                current_input_ids = input_ids
            else:
                current_input_ids = generated_ids[:, -1:]

            position_ids = (
                attention_mask.cumsum(dim=-1) - 1
            )

            if past_key_values is not None:
                position_ids = position_ids[:, -1:]

            kwargs = {
                "input_ids": current_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
            }

            try:
                outputs = self.model(
                    **kwargs,
                    cache_position=torch.arange(
                        attention_mask.shape[-1]
                        - current_input_ids.shape[-1],
                        attention_mask.shape[-1],
                        device=self.device,
                    ),
                )
            except (TypeError, ValueError):
                outputs = self.model(**kwargs)

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token,
                ],
                dim=-1,
            )

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=-1,
            )

            past_key_values = outputs.past_key_values

            if (
                self.tokenizer.eos_token_id is not None
                and torch.all(
                    next_token
                    == self.tokenizer.eos_token_id
                )
            ):
                break

        text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
        if return_diagnostics:
            cache_length = (
                0 if past_key_values is None
                else int(past_key_values.get_seq_length())
                if hasattr(past_key_values, "get_seq_length")
                else int(past_key_values[0][0].shape[-2])
            )
            return GenerationResult(
                text=text,
                generated_token_ids=generated_ids[0, input_ids.shape[-1] :].tolist(),
                cache_length=cache_length,
                retained_positions=list(range(cache_length)),
            )
        return text

    # ------------------------------------------------------------------
    # Cache inspection
    # ------------------------------------------------------------------

    def inspect_cache(
        self,
        past_key_values: Any,
    ) -> Optional[CacheInfo]:
        """
        Inspect the structure and dimensions of a KV cache.
        """

        if past_key_values is None:
            return None

        cache_layers = cache_as_tensor_tuple(past_key_values)

        if len(cache_layers) == 0:
            return CacheInfo(
                num_layers=0,
                sequence_length=0,
                cache_type=type(past_key_values).__name__,
                key_shape=(),
                value_shape=(),
            )

        first_key, first_value = cache_layers[0]

        return CacheInfo(
            num_layers=len(cache_layers),
            sequence_length=first_key.shape[-2],
            cache_type=type(past_key_values).__name__,
            key_shape=tuple(first_key.shape),
            value_shape=tuple(first_value.shape),
        )

    # ------------------------------------------------------------------
    # Stage 2: Attention extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inspect_attention(
        self,
        prompt: str,
    ) -> tuple[
        torch.Tensor,
        list[tuple[int, int]],
    ]:
        """
        Run the model with attentions enabled.

        Returns:
            attentions:
                Tuple-like structure containing attention tensors
                for every layer.

            token_ids:
                Token IDs corresponding to the input sequence.
        """

        inputs = self.tokenize(prompt)

        outputs = self.model(
            **inputs,
            output_attentions=True,
            use_cache=False,
        )

        attentions = outputs.attentions

        token_ids = inputs["input_ids"][0].tolist()

        return (
            attentions,
            [
                (token_id, index)
                for index, token_id in enumerate(token_ids)
            ],
        )

    def analyze_attention(
        self,
        attentions,
    ) -> AttentionAnalysis:
        """
        Aggregate attention across layers and heads.

        For each layer:
            [batch, heads, query, key]

        First average over heads, then average over layers.

        The resulting matrix has shape:
            [sequence_length, sequence_length]
        """

        if attentions is None or len(attentions) == 0:
            raise ValueError(
                "No attention tensors were provided."
            )

        layer_attention = []
        attention_by_layer = []

        for attention in attentions:

            if attention.dim() != 4:
                raise ValueError(
                    "Expected attention tensor with shape "
                    "[batch, heads, query, key]."
                )

            layer = attention[0].detach().float().cpu()
            attention_by_layer.append(layer)
            averaged_heads = layer.mean(dim=0)

            layer_attention.append(
                averaged_heads
            )

        attention_matrix = torch.stack(
            layer_attention,
            dim=0,
        ).mean(dim=0)

        attention_received = attention_matrix.sum(
            dim=0
        )

        early_count = min(
            5,
            attention_matrix.shape[-1],
        )

        early_token_attention = attention_matrix[
            :,
            :early_count,
        ].sum(
            dim=0
        )

        most_attended_token = int(
            torch.argmax(
                attention_received
            ).item()
        )

        return AttentionAnalysis(
            attention_received=attention_received,
            early_token_attention=early_token_attention,
            most_attended_token=most_attended_token,
            attention_matrix=attention_matrix,
            attention_by_layer=tuple(attention_by_layer),
            attention_by_head=torch.stack(attention_by_layer).mean(dim=0),
        )

    def analyze_attention_sinks(
        self,
        attentions,
        token_counts: tuple[int, ...] = (1, 2, 4, 8),
    ) -> dict[str, Any]:
        """
        Analyze how much attention is received by early tokens.

        Returns:
            A dictionary containing:
                - attention_received
                - early_token_attention
                - early_token_fraction
                - most_attended_token
        """

        analysis = self.analyze_attention(
            attentions
        )

        sequence_length = int(analysis.attention_received.shape[-1])
        total_attention = analysis.attention_received.sum().item()
        early_fractions: dict[int, float | None] = {}
        for count in token_counts:
            if count <= 0:
                raise ValueError("token_counts must contain positive integers.")
            if sequence_length < count:
                early_fractions[count] = None
            elif total_attention == 0:
                early_fractions[count] = 0.0
            else:
                early_fractions[count] = float(
                    analysis.attention_received[:count].sum().item()
                    / total_attention
                )

        return {
            "analysis": analysis,
            "sequence_length": sequence_length,
            "attention_received": analysis.attention_received,
            "early_token_attention": early_fractions,
            "most_attended_token": analysis.most_attended_token,
            "most_attended_token_position": analysis.most_attended_token,
        }

    # ------------------------------------------------------------------
    # Stage 3: Sliding-window generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sliding_window_generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        window_size: int = 128,
        return_diagnostics: bool = False,
    ) -> str | GenerationResult:
        """
        Generate tokens while enforcing a fixed-size
        sliding-window KV-cache budget.

        Stage 3:
            - Uses Hugging Face's native DynamicCache.
            - Keeps only the most recent `window_size` entries.
            - Does not convert the cache to the legacy tuple format.

        Stage 6:
            - Tracks absolute token positions independently
              from the physical cache length.
            - Continues using the original sequence positions
              after cache eviction.
        """

        if window_size <= 0:
            raise ValueError(
                "window_size must be positive."
            )

        inputs = self.tokenize(prompt)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        manager = SlidingWindowCacheManager(
            window_size=window_size
        )

        past_key_values = None
        generated_ids = input_ids.clone()
        cache_positions = torch.empty(0, dtype=torch.long)

        # Absolute position in the original sequence.
        #
        # This must not shrink when the physical KV cache
        # is cropped.
        next_position = 0

        for _ in range(max_new_tokens):

            if past_key_values is None:
                current_input_ids = input_ids
            else:
                current_input_ids = generated_ids[:, -1:]

            if past_key_values is None:

                position_ids = build_absolute_position_ids(
                    start_position=0,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

                cache_position = build_cache_position(
                    start_position=0,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

            else:

                position_ids = build_absolute_position_ids(
                    start_position=next_position,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

                cache_position = build_cache_position(
                    start_position=next_position,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

            kwargs = {
                "input_ids": current_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
            }

            try:
                outputs = self.model(
                    **kwargs,
                    cache_position=cache_position,
                )
            except (TypeError, ValueError):
                outputs = self.model(**kwargs)

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token,
                ],
                dim=-1,
            )

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=-1,
            )

            # The next generated token belongs to the next
            # absolute position in the original sequence.
            next_position += current_input_ids.shape[-1]

            # Keep the native Hugging Face DynamicCache.
            past_key_values = outputs.past_key_values
            step_positions = torch.cat(
                (cache_positions, position_ids[0].detach().cpu())
            )

            # Enforce the sliding-window budget using
            # DynamicCache.crop().
            past_key_values = manager.update(
                past_key_values,
                token_positions=step_positions,
            )
            cache_positions = manager.retained_positions

            # The mask must correspond to the retained cache
            # plus the current/new token.
            cache_length = manager.sequence_length(
                past_key_values
            )

            expected_mask_length = cache_length + 1

            if attention_mask.shape[-1] > expected_mask_length:
                attention_mask = attention_mask[
                    :,
                    -expected_mask_length:,
                ]

            if (
                self.tokenizer.eos_token_id is not None
                and torch.all(
                    next_token
                    == self.tokenizer.eos_token_id
                )
            ):
                break

        text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
        if return_diagnostics:
            return GenerationResult(
                text=text,
                generated_token_ids=generated_ids[0, input_ids.shape[-1] :].tolist(),
                cache_length=manager.sequence_length(past_key_values),
                retained_positions=cache_positions.tolist(),
            )
        return text

    # ------------------------------------------------------------------
    # Stage 4: StreamingLLM / attention-sink-aware generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def streaming_llm_generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        cache_budget: int = 128,
        sink_tokens: int = 4,
        return_diagnostics: bool = False,
    ) -> str | GenerationResult:
        """
        Generate tokens using a StreamingLLM-style KV-cache policy.

        The cache retains:
            - the first `sink_tokens` tokens
            - the most recent tokens filling the remaining budget

        Stage 6:
            Absolute position IDs continue increasing after
            cache eviction instead of being recomputed from
            the shortened attention mask.
        """

        if cache_budget <= 0:
            raise ValueError(
                "cache_budget must be positive."
            )

        if sink_tokens < 0:
            raise ValueError(
                "sink_tokens must be non-negative."
            )

        if sink_tokens >= cache_budget:
            raise ValueError(
                "sink_tokens must be smaller than cache_budget."
            )

        inputs = self.tokenize(prompt)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        manager = AttentionSinkCacheManager(
            cache_budget=cache_budget,
            sink_tokens=sink_tokens,
        )

        past_key_values = None
        generated_ids = input_ids.clone()
        cache_positions = torch.empty(0, dtype=torch.long)

        # Absolute position in the original sequence.
        next_position = 0

        for _ in range(max_new_tokens):

            if past_key_values is None:
                current_input_ids = input_ids
            else:
                current_input_ids = generated_ids[:, -1:]

            if past_key_values is None:

                position_ids = build_absolute_position_ids(
                    start_position=0,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

                cache_position = build_cache_position(
                    start_position=0,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

            else:

                position_ids = build_absolute_position_ids(
                    start_position=next_position,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

                cache_position = build_cache_position(
                    start_position=next_position,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

            kwargs = {
                "input_ids": current_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
            }

            try:
                outputs = self.model(
                    **kwargs,
                    cache_position=cache_position,
                )
            except (TypeError, ValueError):
                outputs = self.model(**kwargs)

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token,
                ],
                dim=-1,
            )

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=-1,
            )

            # Continue absolute positions independently
            # from the physical cache length.
            next_position += current_input_ids.shape[-1]

            past_key_values = outputs.past_key_values
            step_positions = torch.cat(
                (cache_positions, position_ids[0].detach().cpu())
            )

            past_key_values = manager.update(
                past_key_values,
                token_positions=step_positions,
            )
            cache_positions = manager.retained_positions

            cache_length = manager.sequence_length(
                past_key_values
            )

            expected_mask_length = cache_length + 1

            if attention_mask.shape[-1] > expected_mask_length:
                attention_mask = attention_mask[
                    :,
                    -expected_mask_length:,
                ]

            if (
                self.tokenizer.eos_token_id is not None
                and torch.all(
                    next_token
                    == self.tokenizer.eos_token_id
                )
            ):
                break

        text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
        if return_diagnostics:
            return GenerationResult(
                text=text,
                generated_token_ids=generated_ids[0, input_ids.shape[-1] :].tolist(),
                cache_length=manager.sequence_length(past_key_values),
                retained_positions=cache_positions.tolist(),
            )
        return text

    # ------------------------------------------------------------------
    # Stage 5: H2O / heavy-hitter generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def heavy_hitter_generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        cache_budget: int = 128,
        sink_tokens: int = 1,
        recent_window: int = 1,
        return_diagnostics: bool = False,
    ) -> str | GenerationResult:
        """
        Generate tokens using an H2O-style heavy-hitter KV-cache policy.

        At every decoding step, attention received by cached tokens
        is accumulated. When the cache exceeds the configured budget,
        the tokens with the highest accumulated attention scores are
        retained.

        Stage 6:
            Absolute position IDs are preserved across
            non-contiguous heavy-hitter eviction.
        """

        if cache_budget <= 0:
            raise ValueError(
                "cache_budget must be positive."
            )

        inputs = self.tokenize(prompt)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        manager = HeavyHitterCacheManager(
            cache_budget=cache_budget,
            sink_tokens=sink_tokens,
            recent_window=recent_window,
        )

        past_key_values = None
        generated_ids = input_ids.clone()

        accumulated_scores = None
        score_updates = 0
        cache_positions = torch.empty(0, dtype=torch.long)

        # Absolute position in the original sequence.
        #
        # Heavy-hitter eviction may remove arbitrary tokens,
        # so this value must remain independent of the
        # physical cache length.
        next_position = 0

        for _ in range(max_new_tokens):

            if past_key_values is None:
                current_input_ids = input_ids
            else:
                current_input_ids = (
                    generated_ids[:, -1:]
                )

            if past_key_values is None:

                position_ids = build_absolute_position_ids(
                    start_position=0,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

                cache_position = build_cache_position(
                    start_position=0,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

            else:

                position_ids = build_absolute_position_ids(
                    start_position=next_position,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

                cache_position = build_cache_position(
                    start_position=next_position,
                    sequence_length=current_input_ids.shape[-1],
                    device=self.device,
                )

            kwargs = {
                "input_ids": current_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "output_attentions": True,
            }

            try:
                outputs = self.model(
                    **kwargs,
                    cache_position=cache_position,
                )
            except (TypeError, ValueError):
                outputs = self.model(**kwargs)

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token,
                ],
                dim=-1,
            )

            # ----------------------------------------------------------
            # Accumulate attention received by each cached token.
            # ----------------------------------------------------------

            if outputs.attentions is None:
                raise RuntimeError(
                    "Stage 5 requires attention tensors, but "
                    "the model did not return them."
                )

            current_attention = aggregate_newest_attention(outputs.attentions)
            accumulated_scores = update_accumulated_scores(
                accumulated_scores, current_attention
            )
            score_updates += 1

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=-1,
            )

            # Continue absolute positions independently
            # from the shortened physical cache.
            next_position += current_input_ids.shape[-1]

            past_key_values = outputs.past_key_values
            step_positions = torch.cat(
                (cache_positions, position_ids[0].detach().cpu())
            )

            # ----------------------------------------------------------
            # Enforce the heavy-hitter cache budget.
            # ----------------------------------------------------------

            past_key_values, accumulated_scores = (
                manager.update(
                    past_key_values,
                    accumulated_scores,
                    token_positions=step_positions,
                )
            )
            cache_positions = manager.retained_positions

            cache_length = manager.sequence_length(
                past_key_values
            )

            expected_mask_length = (
                cache_length + 1
            )

            if (
                attention_mask.shape[-1]
                > expected_mask_length
            ):
                attention_mask = attention_mask[
                    :,
                    -expected_mask_length:,
                ]

            if (
                self.tokenizer.eos_token_id is not None
                and torch.all(
                    next_token
                    == self.tokenizer.eos_token_id
                )
            ):
                break

        text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
        if return_diagnostics:
            return GenerationResult(
                text=text,
                generated_token_ids=generated_ids[0, input_ids.shape[-1] :].tolist(),
                cache_length=manager.sequence_length(past_key_values),
                retained_positions=cache_positions.tolist(),
                score_updates=score_updates,
            )
        return text


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

def main():
    """
    Basic Stage 1/Stage 2 model-wrapper smoke test.
    """

    wrapper = ModelWrapper()

    prompt = (
        "The history of artificial intelligence is"
    )

    print("Model:", wrapper.model_name)
    print("Device:", wrapper.device)
    config = wrapper.model.config
    num_heads = int(config.num_attention_heads)
    head_dim = int(
        getattr(config, "head_dim", config.hidden_size // num_heads)
    )
    print("Transformer layers:", config.num_hidden_layers)
    print("Attention heads:", num_heads)
    print("KV heads:", getattr(config, "num_key_value_heads", num_heads))
    print("Head dimension:", head_dim)

    smoke_inputs = wrapper.tokenize(prompt)
    with torch.no_grad():
        smoke_outputs = wrapper.model(
            **smoke_inputs,
            use_cache=True,
            output_attentions=True,
        )
    cache_info = wrapper.inspect_cache(smoke_outputs.past_key_values)
    if cache_info is not None:
        print("Cache type:", cache_info.cache_type)
        print("Cache layers:", cache_info.num_layers)
        print("Cache sequence length:", cache_info.sequence_length)
        print("Key tensor shape:", cache_info.key_shape)
        print("Value tensor shape:", cache_info.value_shape)
    if smoke_outputs.attentions:
        print(
            "Attention tensor shape:",
            tuple(smoke_outputs.attentions[0].shape),
        )

    print("\nIncremental generation:")

    result = wrapper.incremental_generate(
        prompt,
        max_new_tokens=16,
    )

    print(result)

    print("\nAttention analysis:")

    attentions, token_info = (
        wrapper.inspect_attention(
            prompt
        )
    )

    analysis = wrapper.analyze_attention(
        attentions
    )

    print(
        "Most attended token index:",
        analysis.most_attended_token,
    )

    print(
        "Attention received:",
        analysis.attention_received,
    )

    print("\nAttention sink analysis:")

    sink_analysis = (
        wrapper.analyze_attention_sinks(
            attentions
        )
    )

    print("Attention received by earliest tokens:")
    for count, fraction in sink_analysis["early_token_attention"].items():
        value = "unavailable" if fraction is None else f"{fraction:.6f}"
        print(f"  first {count}: {value}")


if __name__ == "__main__":
    main()