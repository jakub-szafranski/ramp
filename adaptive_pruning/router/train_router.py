"""
Train a compact 4-class router from layer-15 hidden states.

Classes:
- arc (arc_easy + arc_challenge)
- winogrande
- boolq
- hella

Usage:
    python -m adaptive_pruning.router.train_router \
        --input-folder adaptive_pruning/data/router_training \
        --output-model adaptive_pruning/router_model.pt \
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


DEFAULT_LAYER = 15
CLASS_NAMES = ["arc", "winogrande", "boolq", "hella"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _load_hidden(path: Path, layer: int) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    key = f"expert_layer{layer}_last"
    if key not in obj:
        raise KeyError(f"Missing key '{key}' in {path}")
    return obj[key].detach().cpu().numpy().astype(np.float32)


def _stack(files: Sequence[Tuple[str, Path]], layer: int) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for class_name, file_path in files:
        x = _load_hidden(file_path, layer)
        y = np.full(x.shape[0], CLASS_TO_ID[class_name], dtype=np.int64)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 4-class router")
    parser.add_argument("--input-folder", type=Path, required=True, help="Path to folder with training files")
    parser.add_argument("--output-model", type=Path, default="router_model.pt", help="Path to save the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for classifier init")
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help="Hidden layer index used for features (default: 15).",
    )
    args = parser.parse_args()

    root = args.input_folder

    train_files: List[Tuple[str, Path]] = [
        ("arc", root / "hidden_logprobs_wiki_arc_arc_easy_train.pt"),
        ("arc", root / "hidden_logprobs_wiki_arc_arc_challenge_train.pt"),
        ("hella", root / "hidden_logprobs_wiki_hellaswag_hellaswag_train.pt"),
        ("winogrande", root / "hidden_logprobs_wiki_winogrande_winogrande_train.pt"),
        ("boolq", root / "hidden_logprobs_wiki_boolq_boolq_train.pt"),
    ]

    test_groups: Dict[str, List[Path]] = {
        "arc": [
            root / "hidden_logprobs_wiki_arc_arc_easy_test.pt",
            root / "hidden_logprobs_wiki_arc_arc_challenge_test.pt",
        ],
        "winogrande": [root / "hidden_logprobs_wiki_winogrande_winogrande_validation.pt"],
        "hella": [root / "hidden_logprobs_wiki_hellaswag_hellaswag_validation.pt"],
        "boolq": [root / "hidden_logprobs_wiki_boolq_boolq_validation.pt"],
    }

    missing = [p for _, p in train_files if not p.exists()]
    missing += [p for paths in test_groups.values() for p in paths if not p.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing input files:\n{missing_str}")

    x_train, y_train = _stack(train_files, args.layer)

    clf = LogisticRegression(
        solver="saga",  # Supports true multiclass (multinomial) with L1.
        penalty="l1",
        max_iter=10,
        random_state=args.seed,
        verbose=0,
    )
    clf.fit(x_train, y_train)

    print("Per-dataset accuracy:")
    for class_name in CLASS_NAMES:
        files = [(class_name, p) for p in test_groups[class_name]]
        x_test, y_test = _stack(files, args.layer)
        y_pred = clf.predict(x_test)
        print(f"- {class_name}: {_acc(y_test, y_pred):.3f} ({len(y_test)} samples)")

    print("Saving model...")
    torch.save(clf, args.output_model)

if __name__ == "__main__":
    main()
