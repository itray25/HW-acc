import argparse
import math
import time
from contextlib import nullcontext
from typing import Iterable, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from SnapKV import SnapKVPress


def patch_gpt_neox_for_kvpress(model):
    if not hasattr(model, "gpt_neox"):
        return

    if not hasattr(model, "model"):
        model.model = model.gpt_neox

    if not hasattr(model.gpt_neox, "rotary_emb"):
        attn = getattr(model.gpt_neox.layers[0], "attention", None)
        if attn and hasattr(attn, "rotary_emb"):
            model.gpt_neox.rotary_emb = attn.rotary_emb

    for i, layer in enumerate(model.gpt_neox.layers):
        if not hasattr(layer, "self_attn") and hasattr(layer, "attention"):
            layer.self_attn = layer.attention
        if hasattr(layer, "self_attn"):
            if not hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = i
            if not hasattr(layer.self_attn, "head_dim") and hasattr(
                layer.self_attn, "head_size"
            ):
                layer.self_attn.head_dim = layer.self_attn.head_size


def iter_chunks(token_ids: torch.Tensor, block_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, token_ids.size(0), block_size):
        chunk = token_ids[start : start + block_size]
        if chunk.numel() > 1:
            yield chunk


def compute_ppl(model, tokenizer, texts, block_size, press, device):
    total_nll, total_tokens = 0.0, 0
    model.eval()

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[
            0
        ].to(device)
        cache = DynamicCache()
        for chunk in iter_chunks(ids, block_size):
            inp = chunk.unsqueeze(0)
            ctx = press(model) if press else nullcontext()
            with torch.no_grad(), ctx:
                out = model(input_ids=inp, past_key_values=cache, use_cache=True)
            logits = out.logits[:, :-1, :]
            labels = inp[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum"
            )
            total_nll += loss.item()
            total_tokens += labels.numel()

    return math.exp(total_nll / max(total_tokens, 1))


def benchmark_speed(model, tokenizer, texts, block_size, press, device):
    model.eval()
    total_tokens = 0
    start = time.perf_counter()

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[
            0
        ].to(device)
        cache = DynamicCache()
        for chunk in iter_chunks(ids, block_size):
            inp = chunk.unsqueeze(0)
            ctx = press(model) if press else nullcontext()
            with torch.no_grad(), ctx:
                model(input_ids=inp, past_key_values=cache, use_cache=True)
            total_tokens += inp.numel()

    elapsed = max(time.perf_counter() - start, 1e-8)
    return total_tokens / elapsed


def load_texts(dataset_name, split, max_samples, pg19_sample_id):
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts = [row["text"] for row in ds if row["text"].strip()]
    elif dataset_name == "pg19":
        ds = load_dataset("pg19", split=split)
        texts = [ds[pg19_sample_id]["text"]]
    else:
        raise ValueError("dataset_name must be wikitext or pg19")
    return texts[:max_samples]


def build_press(args):
    if args.press == "none":
        return None
    if args.press == "snapkv":
        return SnapKVPress(
            compression_ratio=args.compression_ratio,
            window_size=args.window_size,
            kernel_size=args.kernel_size,
        )
    raise ValueError("Unsupported press")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", default="wikitext", choices=["wikitext", "pg19"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--pg19-sample-id", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--press", default="snapkv", choices=["none", "snapkv"])
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--mode", default="ppl", choices=["ppl", "speed"])
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    patch_gpt_neox_for_kvpress(model)

    texts = load_texts(args.dataset, args.split, args.max_samples, args.pg19_sample_id)
    press = build_press(args)

    if args.mode == "ppl":
        ppl = compute_ppl(model, tokenizer, texts, args.block_size, press, device)
        print(f"PPL ({args.dataset}, press={args.press}): {ppl:.4f}")
    else:
        tps = benchmark_speed(model, tokenizer, texts, args.block_size, press, device)
        print(f"Speed ({args.dataset}, press={args.press}): {tps:.2f} tokens/s")


if __name__ == "__main__":
    main()
