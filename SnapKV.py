import math
from dataclasses import dataclass
import importlib
from typing import Any

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states, extract_keys_and_values


def _torch():
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    F = importlib.import_module("torch.nn.functional")
    llama = importlib.import_module("transformers.models.llama.modeling_llama")
    return torch, nn, F, llama.repeat_kv, llama.rotate_half


def _prerope_queries(module, hidden_states):
    try:
        return get_prerope_query_states(module, hidden_states)
    except NotImplementedError:
        if not (hasattr(module, "query_key_value") and hasattr(module, "head_size")):
            raise
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        qkv = module.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, num_heads, 3 * module.head_size).transpose(1, 2)
        query_states, _, _ = qkv.chunk(3, dim=-1)
        return query_states


@dataclass
class SnapKVPress(ScorerPress):
    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5

    @staticmethod
    def compute_window_attention(
        module, hidden_states, keys, window_size, position_embeddings
    ):
        torch, nn, _, repeat_kv, rotate_half = _torch()

        bsz, _, k_len, _ = keys.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_kv_heads = getattr(module.config, "num_key_value_heads", num_heads)
        num_groups = num_heads // num_kv_heads

        q = _prerope_queries(module, hidden_states[:, -window_size:])

        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        if cos.shape[-1] != head_dim:
            r = cos.shape[-1]
            q_rot, q_pass = q[..., :r], q[..., r:]
            q_rot = q_rot * cos.unsqueeze(1) + rotate_half(q_rot) * sin.unsqueeze(1)
            q = torch.cat([q_rot, q_pass], dim=-1)
        else:
            q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)

        k = repeat_kv(keys, num_groups)
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)

        mask = torch.ones_like(attn) * float("-inf")
        mask = torch.triu(mask, diagonal=k_len - window_size + 1)
        attn += mask

        attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        return attn[..., :-window_size]

    def forward_hook(self, module, input, kwargs, output):
        torch, _, _, _, _ = _torch()
        try:
            from transformers import QuantizedCache
        except Exception:
            QuantizedCache = None

        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and input:
            hidden_states = input[0]

        cache = kwargs.get("past_key_values") or kwargs.get("layer_past")
        cache_position = kwargs.get("cache_position")

        if hidden_states is None or cache is None or cache_position is None:
            return output

        q_len = hidden_states.shape[1]
        if q_len <= 1 or module.layer_idx >= len(cache.layers):
            return output

        cache_layer = cache.layers[module.layer_idx]
        keys, values = extract_keys_and_values(cache, module.layer_idx)
        keys, values = self.compress(
            module, hidden_states, keys, values, output[1], kwargs
        )

        if QuantizedCache is not None and isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(
                keys, axis=cache_layer.axis_key
            )
            cache_layer._quantized_values = cache_layer._quantize(
                values, axis=cache_layer.axis_value
            )
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values

        return output

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        torch, _, F, _, _ = _torch()

        bsz, num_kv_heads, k_len, _ = keys.shape
        num_groups = module.config.num_attention_heads // num_kv_heads

        if hidden_states.shape[1] <= self.window_size:
            return torch.ones(
                (bsz, num_kv_heads, k_len), device=keys.device, dtype=keys.dtype
            )

        if attentions is not None:
            attn = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn = self.compute_window_attention(
                module,
                hidden_states,
                keys,
                self.window_size,
                kwargs["position_embeddings"],
            )

        scores = attn.mean(dim=-2)
        scores = F.avg_pool1d(
            scores,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )
        scores = scores.view(
            bsz, num_kv_heads, num_groups, k_len - self.window_size
        ).mean(2)
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())

        return scores
