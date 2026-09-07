"""
Stage 1: Qwen2.5 model loading and KV-cache inspection.

This module intentionally does not implement cache eviction.
It provides:
    - tokenizer/model loading
    - automatic CPU/CUDA selection
    - a normal forward pass
    - incremental generation using past_key_values
    - KV-cache inspection
    - attention tensor inspection when supported
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@dataclass
class CacheInfo:
    """Summary of the model's KV cache."""

    cache_type: str
    num_layers: int
    sequence_length: int
    key_shape: tuple[int, ...] | None
    value_shape: tuple[int, ...] | None


class QwenModelWrapper:
    """Small wrapper around Qwen2.5 for Task 3 experiments."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = self._select_device(device)

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading model on {self.device}...")

        # Eager attention is deliberately selected because Stage 2 will
        # inspect attention probabilities. Some optimized attention
        # implementations do not return attention weights.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="eager",
            )
        except TypeError:
            # Compatibility with older Transformers releases.
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _select_device(device: str | None) -> torch.device:
        """Select CPU/CUDA automatically unless explicitly requested."""

        if device is not None:
            requested = torch.device(device)

            if requested.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but CUDA is not available."
                )

            return requested

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenize text and move tensors to the selected device."""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )

        return {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

    def forward(
        self,
        text: str,
        output_attentions: bool = True,
    ) -> Any:
        """
        Perform a normal forward pass.

        Returns the raw Hugging Face model output so later stages can
        directly access logits, attentions and past_key_values.
        """

        inputs = self.tokenize(text)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=True,
                output_attentions=output_attentions,
                return_dict=True,
            )

        return outputs

    @staticmethod
    def _cache_type(cache: Any) -> str:
        """Return a readable cache type name."""

        if cache is None:
            return "None"

        return type(cache).__name__

    @staticmethod
    def _get_cache_layer(
        cache: Any,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return key/value tensors for one layer.

        Supports both:
            1. modern Hugging Face Cache objects
            2. legacy tuple-style past_key_values
        """

        # Modern Cache API.
        if hasattr(cache, "layers"):
            layer = cache.layers[layer_index]

            key_cache = getattr(layer, "keys", None)
            value_cache = getattr(layer, "values", None)

            if key_cache is not None and value_cache is not None:
                return key_cache, value_cache

            # Some older DynamicCache versions expose key/value through
            # key_cache/value_cache on the cache itself.
            if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
                return (
                    cache.key_cache[layer_index],
                    cache.value_cache[layer_index],
                )

        # Older DynamicCache implementation.
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            return (
                cache.key_cache[layer_index],
                cache.value_cache[layer_index],
            )

        # Legacy tuple:
        # past_key_values[layer] = (key_states, value_states)
        if isinstance(cache, (tuple, list)):
            layer = cache[layer_index]

            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                return layer[0], layer[1]

        raise TypeError(
            f"Unsupported past_key_values format: {type(cache)!r}"
        )

    @classmethod
    def get_cache_sequence_length(cls, cache: Any) -> int:
        """Return the number of cached tokens."""

        if cache is None:
            return 0

        # Modern Hugging Face Cache API.
        if hasattr(cache, "get_seq_length"):
            try:
                return int(cache.get_seq_length())
            except TypeError:
                pass

        # Legacy tuple-style cache.
        key, _ = cls._get_cache_layer(cache, 0)

        return int(key.shape[-2])

    @classmethod
    def inspect_cache(cls, cache: Any) -> CacheInfo:
        """Inspect the structure and tensor dimensions of a KV cache."""

        if cache is None:
            return CacheInfo(
                cache_type="None",
                num_layers=0,
                sequence_length=0,
                key_shape=None,
                value_shape=None,
            )

        if hasattr(cache, "layers"):
            num_layers = len(cache.layers)
        elif hasattr(cache, "key_cache"):
            num_layers = len(cache.key_cache)
        elif isinstance(cache, (tuple, list)):
            num_layers = len(cache)
        else:
            raise TypeError(
                f"Unsupported past_key_values format: {type(cache)!r}"
            )

        key, value = cls._get_cache_layer(cache, 0)

        return CacheInfo(
            cache_type=cls._cache_type(cache),
            num_layers=num_layers,
            sequence_length=cls.get_cache_sequence_length(cache),
            key_shape=tuple(key.shape),
            value_shape=tuple(value.shape),
        )

    def inspect_attention(
        self,
        attentions: Any,
    ) -> list[tuple[int, tuple[int, ...]]]:
        """
        Return attention tensor dimensions for every layer.

        For a causal decoder these are normally:

            [batch, num_heads, query_length, key_length]

        Stage 2 will perform actual attention aggregation and analysis.
        """

        if attentions is None:
            return []

        result: list[tuple[int, tuple[int, ...]]] = []

        for layer_index, attention in enumerate(attentions):
            if attention is None:
                continue

            result.append(
                (
                    layer_index,
                    tuple(attention.shape),
                )
            )

        return result

    def incremental_generate(
        self,
        prompt: str,
        max_new_tokens: int = 8,
        collect_attentions: bool = True,
    ) -> dict[str, Any]:
        """
        Generate tokens one at a time using past_key_values.

        This deliberately performs the decoding loop manually rather than
        using model.generate(), because Task 3 needs direct visibility
        into the KV cache.
        """

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        inputs = self.tokenize(prompt)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # --------------------------------------------------------------
        # Prefill
        # --------------------------------------------------------------
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=collect_attentions,
                return_dict=True,
            )

        past_key_values = outputs.past_key_values

        generated_ids = input_ids.clone()

        attention_history: list[Any] = []

        if collect_attentions and outputs.attentions is not None:
            attention_history.append(outputs.attentions)

        # --------------------------------------------------------------
        # Incremental decoding
        # --------------------------------------------------------------
        for _ in range(max_new_tokens):
            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=-1,
            )

            past_length = self.get_cache_sequence_length(
                past_key_values
            )

            # The new token is at the next original position.
            position_ids = torch.tensor(
                [[past_length]],
                dtype=torch.long,
                device=self.device,
            )

            # The attention mask covers cached tokens + current token.
            next_attention_mask = torch.ones(
                (input_ids.shape[0], past_length + 1),
                dtype=torch.long,
                device=self.device,
            )

            model_kwargs = {
                "input_ids": next_token,
                "attention_mask": next_attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "output_attentions": collect_attentions,
                "return_dict": True,
            }

            # Recent Transformers versions accept cache_position.
            # Older versions may not, so retry without it.
            cache_position = torch.tensor(
                [past_length],
                dtype=torch.long,
                device=self.device,
            )

            model_kwargs["cache_position"] = cache_position

            try:
                with torch.no_grad():
                    outputs = self.model(**model_kwargs)
            except TypeError as exc:
                if "cache_position" not in str(exc):
                    raise

                model_kwargs.pop("cache_position")

                with torch.no_grad():
                    outputs = self.model(**model_kwargs)

            past_key_values = outputs.past_key_values

            if collect_attentions and outputs.attentions is not None:
                attention_history.append(outputs.attentions)

        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return {
            "generated_ids": generated_ids,
            "generated_text": generated_text,
            "past_key_values": past_key_values,
            "attention_history": attention_history,
        }

    def print_model_summary(self) -> None:
        """Print the model architecture information relevant to Task 3."""

        config = self.model.config

        num_layers = getattr(config, "num_hidden_layers", "unknown")
        num_heads = getattr(config, "num_attention_heads", "unknown")
        num_kv_heads = getattr(
            config,
            "num_key_value_heads",
            num_heads,
        )

        head_dim = getattr(config, "head_dim", None)

        if head_dim is None:
            hidden_size = getattr(config, "hidden_size", None)

            if hidden_size is not None and isinstance(num_heads, int):
                head_dim = hidden_size // num_heads

        print()
        print("=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)
        print(f"Model:                 {self.model_name}")
        print(f"Device:                {self.device}")
        print(f"Parameters:            {self.model.num_parameters():,}")
        print(f"Layers:                {num_layers}")
        print(f"Attention heads:       {num_heads}")
        print(f"KV heads:              {num_kv_heads}")
        print(f"Head dimension:        {head_dim}")
        print(
            f"Max position embeddings: "
            f"{getattr(config, 'max_position_embeddings', 'unknown')}"
        )
        print(f"Model dtype:           {self.model.dtype}")
        print("=" * 70)

    def run_smoke_test(self) -> None:
        """
        Run the Stage 1 smoke test.

        The smoke test:
            1. loads the model;
            2. performs a forward pass;
            3. inspects the KV cache;
            4. inspects attention dimensions;
            5. performs incremental generation.
        """

        prompt = (
            "Attention mechanisms allow transformer models to "
            "process information across a sequence."
        )

        self.print_model_summary()

        # --------------------------------------------------------------
        # Forward pass
        # --------------------------------------------------------------
        print("\n[1/3] Running forward pass...")

        outputs = self.forward(
            prompt,
            output_attentions=True,
        )

        cache_info = self.inspect_cache(
            outputs.past_key_values
        )

        print("Forward pass: PASS")
        print(f"Cache type:             {cache_info.cache_type}")
        print(f"Cache layers:           {cache_info.num_layers}")
        print(f"Cached sequence length: {cache_info.sequence_length}")
        print(f"First-layer key shape:  {cache_info.key_shape}")
        print(f"First-layer value shape:{cache_info.value_shape}")

        # --------------------------------------------------------------
        # Attention inspection
        # --------------------------------------------------------------
        print("\n[2/3] Inspecting attention tensors...")

        attention_info = self.inspect_attention(
            outputs.attentions
        )

        if attention_info:
            print(
                f"Attention tensors found: "
                f"{len(attention_info)} layers"
            )

            for layer_index, shape in attention_info[:3]:
                print(
                    f"  Layer {layer_index}: {shape}"
                )

            if len(attention_info) > 3:
                print("  ...")
        else:
            print(
                "Attention tensors were not returned by this "
                "Transformers/model configuration."
            )

        # --------------------------------------------------------------
        # Incremental generation
        # --------------------------------------------------------------
        print("\n[3/3] Running incremental generation...")

        result = self.incremental_generate(
            prompt,
            max_new_tokens=4,
            collect_attentions=True,
        )

        final_cache_info = self.inspect_cache(
            result["past_key_values"]
        )

        print("Incremental generation: PASS")
        print(
            f"Generated text: "
            f"{result['generated_text']}"
        )
        print(
            f"Final cache sequence length: "
            f"{final_cache_info.sequence_length}"
        )
        print(
            f"Attention snapshots collected: "
            f"{len(result['attention_history'])}"
        )

        print("\n" + "=" * 70)
        print("STAGE 1 SMOKE TEST: PASS")
        print("=" * 70)


def main() -> None:
    """Entry point for `python -m src.model_wrapper`."""

    wrapper = QwenModelWrapper()
    wrapper.run_smoke_test()


if __name__ == "__main__":
    main()