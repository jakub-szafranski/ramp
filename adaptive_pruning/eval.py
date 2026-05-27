"""
Evaluate Dense / FLAP / WandaSP on lm-evaluation-harness benchmarks.

Supports three modes:
  dense   – base model, no pruning
  flap    – static FLAP expert applied to all samples
  wandasp – static WandaSP expert applied to all samples

Usage
-----
    # Static FLAP expert
    python -m adaptive_pruning.eval \
        --llm huggyllama/llama-7b \
        --mode flap \
        --expert mom/experts/wiki02_llama7b.pt \
        --tasks boolq arc_easy hellaswag \
        --limit 200 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer

from adaptive_pruning.utils.prunable_llm import PrunableLLM, load_llm
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM



def _write_log(args: argparse.Namespace, results: dict) -> None:
    if not args.log_file:
        return
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "llm": args.llm,
        "mode": args.mode,
        "expert": args.expert,
        "tasks": args.tasks,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "results": results.get("results", {}),
    }
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def evaluate(args: argparse.Namespace) -> dict:
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    print(f"Loading LLM: {args.llm} …")
    model = load_llm(args.llm)
    model.to(args.device).eval()

    llm_tok = AutoTokenizer.from_pretrained(args.llm)
    llm_tok.pad_token = llm_tok.eos_token
    llm_tok.padding_side = "left"

    wrapper = PrunableLLM(model, device=args.device)

    # ── mode-specific setup ───────────────────────────────────────────
    if args.mode == "dense":
        print("Mode: dense (no pruning)")
        eval_model = wrapper.model

    elif args.mode in ("flap", "wandasp"):
        assert args.expert, f"--expert required for mode={args.mode}"
        print(f"Mode: {args.mode} (static expert: {args.expert})")
        expert = torch.load(args.expert, map_location="cpu", weights_only=False)
        wrapper.prune(expert, unstr=False)  # structured for real eval
        eval_model = wrapper.model

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    lm = HFLM(
        pretrained=eval_model,
        tokenizer=llm_tok,
        batch_size=args.batch_size,
    )

    # ── run evaluation ────────────────────────────────────────────────
    print(f"Tasks: {args.tasks}")
    if args.limit:
        print(f"Limit: {args.limit} samples per task")

    results = simple_evaluate(
        model=lm,
        tasks=args.tasks,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    # ── print results ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}")

    for task, metrics in results.get("results", {}).items():
        print(f"\n  {task}:")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"    {k:30s} {v:.4f}")
            elif k != "alias":
                print(f"    {k:30s} {v}")

    # ── cleanup static pruning ────────────────────────────────────────
    if args.mode in ("flap", "wandasp") and wrapper.is_pruned:
        wrapper.unprune()

    _write_log(args, results)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate dense / FLAP / WandaSP on lm-eval-harness"
    )
    ap.add_argument("--llm", default="huggyllama/llama-7b")
    ap.add_argument(
        "--mode",
        choices=["dense", "flap", "wandasp"],
        required=True,
        help="Evaluation mode.",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["boolq", "arc_easy", "arc_challenge", "winogrande", "hellaswag"],
        help="lm-eval task names (e.g. boolq arc_easy hellaswag).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per task (None = full benchmark).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for lm-eval (default: 256).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducibility/logging.",
    )
    ap.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Append JSONL logs of each run to this file.",
    )
    ap.add_argument("--expert", default=None, help="Path to .pt expert for flap/wandasp mode.")

    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
