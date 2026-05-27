#!/usr/bin/env python3
"""Collect prompt-only hidden states for one model mode.

Modes:
- dense: no pruning
- pruned: prune once to provided expert and collect on that fixed subnet

For each sample, collects:
- last-token hidden state for requested layers (prompt-only context)

Input data is a .pt list of standardized rows from adaptive_pruning.router.collect_training_data.

Usage
-----
    python -m adaptive_pruning.router.collect_hidden \
        --expert dense \
        --model huggyllama/llama-7b \
        --data adaptive_pruning/data/arc_easy_validation.pt \
        --limit 1000 \
        --device cuda:0 \
        --output adaptive_pruning/data/hidden_states_arc_easy_val.pt \
        --seed 42 \
        --batch_size 256 \
        --hidden_layers 15
"""

from __future__ import annotations

import argparse
import os
import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from models.hf_llama.modeling_llama import LlamaForCausalLM
from adaptive_pruning.utils.prunable_llm import PrunableLLM


def get_llm(model_name: str, device: str = "cuda:0", cache_dir: str = "llm_weights"):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map=device,
    )
    return model


@torch.no_grad()
def collect_batch(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    hidden_layer_indices: list[int] = (4,),
):
    """Collect prompt-only last-token hidden states for a batch."""
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    feat_out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)

    hidden_dict = {}
    for li in hidden_layer_indices:
        hs = feat_out.hidden_states[li + 1].float()
        last_tok = hs[:, -1, :].cpu()
        hidden_dict[f"layer{li}_last"] = last_tok

    return hidden_dict


def _validate_sample_schema(sample: dict, idx: int):
    required = ("prompt",)
    missing = [k for k in required if k not in sample]
    if missing:
        raise ValueError(f"Sample {idx} missing keys: {missing}")

    if not isinstance(sample["prompt"], str):
        raise ValueError(f"Sample {idx} has non-string prompt")


def main():
    parser = argparse.ArgumentParser(
        description="Collect prompt-only last-token hidden states for one mode"
    )
    parser.add_argument(
        "--expert",
        type=str,
        required=True,
        help="Use 'dense' for unpruned model, otherwise path to expert .pt file",
    )
    parser.add_argument("--model", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--data", type=str, required=True, help="Path to standardized .pt data file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="adaptive_pruning/data/hidden_states.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--hidden_layers", type=int, nargs="+", default=[15],
        help="Layer indices to extract last-token hidden states from",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.expert != "dense" and not os.path.exists(args.expert):
        raise FileNotFoundError(f"Expert file not found: {args.expert}")

    print(f"[collect] Loading data: {args.data}")
    data = torch.load(args.data, weights_only=False)
    if not isinstance(data, list):
        raise ValueError("Input --data must be a list of sample dicts")

    if args.limit and args.limit < len(data):
        data = random.sample(data, args.limit)

    for idx, sample in enumerate(data):
        _validate_sample_schema(sample, idx)

    n = len(data)
    print(f"  {n} samples")

    print(f"[collect] Loading model: {args.model}")
    model = get_llm(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    wrapper = PrunableLLM(model, device=args.device)
    base_sparsity = wrapper.sparsity()
    print(f"[collect] Model sparsity after load: {base_sparsity:.6f} ({base_sparsity * 100:.2f}%)")

    if args.expert == "dense":
        print("[collect] Mode: dense (no pruning)")
    else:
        print(f"[collect] Mode: pruned expert -> {args.expert}")
        expert = torch.load(args.expert, map_location="cpu", weights_only=False)
        wrapper.prune(expert, unstr=False)
        print("[collect] Expert pruning applied")
        pruned_sparsity = wrapper.sparsity()
        print(f"[collect] Model sparsity after pruning: {pruned_sparsity:.6f} ({pruned_sparsity * 100:.2f}%)")

    hidden_layer_indices = args.hidden_layers
    n_layers = int(model.config.num_hidden_layers)
    bad_layers = [li for li in hidden_layer_indices if li < 0 or li >= n_layers]
    if bad_layers:
        raise ValueError(f"Invalid hidden layer indices {bad_layers}; valid range is [0, {n_layers - 1}]")

    results = {}
    for li in hidden_layer_indices:
        results[f"expert_layer{li}_last"] = []

    print("\n[collect] Processing samples...")
    batch_size = max(int(args.batch_size), 1)
    for start in tqdm(range(0, n, batch_size), desc="Collect"):
        batch = data[start:start + batch_size]
        prompts = [s["prompt"] for s in batch]

        out = collect_batch(
            model, tokenizer, prompts,
            args.device, hidden_layer_indices,
        )
        for li in hidden_layer_indices:
            results[f"expert_layer{li}_last"].append(out[f"layer{li}_last"])

    if args.expert != "dense":
        wrapper.unprune()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[collect] Saving results...")
    save_dict = {
        "hidden_layers": hidden_layer_indices,
        "feature_context": "prompt",
        "expert": args.expert,
        "n_samples": n,
    }
    for li in hidden_layer_indices:
        stacked = torch.cat(results[f"expert_layer{li}_last"], dim=0)
        if stacked.shape[0] != n:
            raise RuntimeError(
                f"Hidden layer {li} sample count mismatch: got {stacked.shape[0]}, expected {n}"
            )
        save_dict[f"expert_layer{li}_last"] = stacked

    torch.save(save_dict, args.output)
    print(f"  Saved to {args.output}")
    print("  Hidden layer shapes:")
    for li in hidden_layer_indices:
        shape = save_dict[f"expert_layer{li}_last"].shape
        print(f"    layer{li}_last: {tuple(shape)}")
    print("Done.")


if __name__ == "__main__":
    main()
