"""Reformat benchmark MCQ data into router-training .pt files (no LLM).

Usage:
    python -m adaptive_pruning.router.collect_training_data \
            --dataset arc_easy \
            --output adaptive_pruning/data/arc_easy

Output files follow: <output>_<split>.pt
"""

from __future__ import annotations

import argparse
import os
import re

import torch
from datasets import get_dataset_split_names, load_dataset
from tqdm import tqdm

TASK_DEFAULT_SPLIT = {
    "boolq": "validation",
    "winogrande": "test",
    "hellaswag": "test",
    "arc_easy": "test",
    "arc_challenge": "test",
    "openbookqa": "test",
    "math_qa": "test",
}

TASK_OUTPUT_SPLITS = {
    "boolq": ["train", "validation"],
    "winogrande": ["train", "validation", "test"],
    "hellaswag": ["train", "validation", "test"],
    "arc_easy": ["train", "validation", "test"],
    "arc_challenge": ["train", "validation", "test"],
    "openbookqa": ["train", "validation", "test"],
    "math_qa": ["train", "val", "test"],
}

TASK_HF = {
    "boolq": ("google/boolq", None),
    "winogrande": ("allenai/winogrande", "winogrande_xl"),
    "hellaswag": ("Rowan/hellaswag", None),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge"),
    "openbookqa": ("allenai/openbookqa", "main"),
    "math_qa": ("regisss/math_qa", None),
}

TASK_SPLIT_ALIASES = {
    "math_qa": {
        "val": ["validation", "validation"],
    },
}


def _preprocess_hellaswag(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ").strip()


def _arc_label_map() -> dict[str, int]:
    out = {letter: i for i, letter in enumerate("ABCDE")}
    out.update({str(i + 1): i for i in range(5)})
    return out


def _mathqa_option_list(options: str) -> list[str]:
    pattern = re.compile(r"([a-e])\s*\)\s*(.*?)(?=(?:,\s*[a-e]\s*\))|$)")
    return [match.group(2).strip() for match in pattern.finditer(options)]


def _resolve_split(task: str, split: str, available: set[str]) -> str:
    candidates = TASK_SPLIT_ALIASES.get(task, {}).get(split, [split])
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(
        f"Task '{task}' does not have a usable split for '{split}'. Available: {sorted(available)}"
    )


def _iter_formatted(task: str, split: str):
    available = set(get_dataset_split_names(*TASK_HF[task]))
    split = _resolve_split(task, split, available)

    if task == "boolq":
        ds = load_dataset("google/boolq", split=split)
        for row in ds:
            yield {
                "question": row["question"],
                "answers": ["Yes", "No"],
                "correct": 0 if row["answer"] else 1,
                "prompt": row["passage"] + "\nQuestion: " + row["question"] + "?\nAnswer:",
                "choices": [" yes", " no"],
                "_src": "boolq",
            }
        return

    if task == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split=split)
        for row in ds:
            if row["answer"] not in ("1", "2"):
                continue
            parts = row["sentence"].split("_")
            rest = parts[1] if len(parts) > 1 else ""
            yield {
                "question": row["sentence"],
                "answers": [row["option1"], row["option2"]],
                "correct": int(row["answer"]) - 1,
                "prompt": parts[0],
                "choices": [row["option1"] + rest, row["option2"] + rest],
                "_src": "winogrande",
            }
        return

    if task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split=split)
        for row in ds:
            endings = row["endings"]
            if len(endings) != 4:
                continue
            label = row["label"]
            if isinstance(label, str):
                if label not in ("0", "1", "2", "3"):
                    continue
                label = int(label)
            ctx = row["ctx_a"] + " " + row["ctx_b"].rstrip()
            yield {
                "question": row["ctx"],
                "answers": list(endings),
                "correct": int(label),
                "prompt": _preprocess_hellaswag(row["activity_label"] + ": " + ctx),
                "choices": [" " + _preprocess_hellaswag(e) for e in endings],
                "_src": "hellaswag",
            }
        return

    if task in ("arc_easy", "arc_challenge"):
        cfg = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", cfg, split=split)
        label_map = _arc_label_map()
        for row in ds:
            answer_texts = row["choices"]["text"]
            key = row.get("answerKey", "")
            if key not in label_map:
                continue
            gold = label_map[key]
            if gold >= len(answer_texts):
                continue
            yield {
                "question": row["question"],
                "answers": answer_texts,
                "correct": gold,
                "prompt": "Question: " + row["question"] + "\nAnswer:",
                "choices": [" " + t for t in answer_texts],
                "_src": task,
            }
        return

    if task == "openbookqa":
        ds = load_dataset("allenai/openbookqa", "main", split=split)
        label_map = _arc_label_map()
        for row in ds:
            answer_texts = row["choices"]["text"]
            key = row.get("answerKey", "")
            if key not in label_map:
                continue
            gold = label_map[key]
            if gold >= len(answer_texts):
                continue
            yield {
                "question": row["question_stem"],
                "answers": answer_texts,
                "correct": gold,
                "prompt": "Question: " + row["question_stem"] + "\nAnswer:",
                "choices": [" " + t for t in answer_texts],
                "_src": "openbookqa",
            }
        return

    if task == "math_qa":
        ds = load_dataset("regisss/math_qa", split=split)
        for row in ds:
            options = _mathqa_option_list(row["options"])
            correct = row["correct"].strip().lower()
            correct_idx = ord(correct) - ord("a")
            if not 0 <= correct_idx < len(options):
                continue
            yield {
                "question": row["Problem"],
                "answers": options,
                "correct": correct_idx,
                "prompt": "Question: " + row["Problem"] + "\nAnswer:",
                "choices": [" " + t for t in options],
                "_src": "math_qa",
            }
        return

    raise ValueError(f"Unsupported task: {task}")


def _validate_splits(task: str):
    name, cfg = TASK_HF[task]
    available = set(get_dataset_split_names(name, cfg))
    required = TASK_OUTPUT_SPLITS[task]
    missing = []
    for split in required:
        try:
            _resolve_split(task, split, available)
        except ValueError:
            missing.append(split)
    if missing:
        raise ValueError(
            f"Task '{task}' is missing required split(s): {missing}. "
            f"Available: {sorted(available)}"
        )


def _save_for_split(task: str, split: str, output_prefix: str) -> str:
    rows = list(tqdm(_iter_formatted(task, split), desc=f"{task}:{split}"))
    out_path = f"{output_prefix}_{split}.pt"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(rows, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Reformat benchmark data to .pt (no LLM)")
    parser.add_argument("--dataset", required=True, choices=sorted(TASK_OUTPUT_SPLITS.keys()))
    parser.add_argument(
        "--output",
        required=True,
        help="Output path prefix. Script writes <output>_<split>.pt",
    )
    args = parser.parse_args()

    task = args.dataset
    output_prefix = args.output[:-3] if args.output.endswith(".pt") else args.output

    print(f"Dataset: {task}")
    print(f"Default split (hardcoded): {TASK_DEFAULT_SPLIT[task]}")
    print(f"Will export splits: {TASK_OUTPUT_SPLITS[task]}")

    _validate_splits(task)

    for split in TASK_OUTPUT_SPLITS[task]:
        out_path = _save_for_split(task, split, output_prefix)
        print(f"Saved {split}: {out_path}")


if __name__ == "__main__":
    main()
