import argparse
import copy
import torch

def blend_experts(expert1_path: str, expert2_path: str, split_layer: int, out_path: str):
    """
    Blend two expert models by combining layers from each based on a split point.

    This function creates a new model by taking layers from expert1 up to and including
    the split_layer, and layers from expert2 after the split_layer. The resulting blended
    model is saved to the specified output path.

    Args:
        expert1_path (str): Path to the first expert model checkpoint.
        expert2_path (str): Path to the second expert model checkpoint.
        split_layer (int): The layer index that determines the split point.
                          Layers with indices <= split_layer are taken from expert1,
                          layers with indices > split_layer are taken from expert2.
        out_path (str): Path where the blended model checkpoint will be saved.

    Returns:
        str: The output path where the blended model was saved.

    Raises:
        ValueError: If the two expert models have no common layers.

    Note:
        - Both expert models must have a "layers" key in their checkpoint dictionaries.
        - The function performs deep copies of all layers to avoid unintended mutations.
        - Metadata about the blending operation (expert paths, split layer, blending rule)
          is stored in the output checkpoint under the "mix_info" key.
    """
    e1 = torch.load(expert1_path, map_location="cpu", weights_only=False)
    e2 = torch.load(expert2_path, map_location="cpu", weights_only=False)

    l1, l2 = e1["layers"], e2["layers"]
    keys = sorted(set(l1.keys()) & set(l2.keys()), key=int)
    if not keys:
        raise ValueError("No common layers between experts.")

    mixed_layers = {}
    for k in keys:
        src = l1 if int(k) <= split_layer else l2
        mixed_layers[k] = copy.deepcopy(src[k])

    out = {k: copy.deepcopy(v) for k, v in e1.items() if k != "layers"}
    out["layers"] = mixed_layers
    out["mix_info"] = {
        "expert1": expert1_path,
        "expert2": expert2_path,
        "split_layer": split_layer,
        "rule": "<= split_layer from expert1, > split_layer from expert2",
    }

    torch.save(out, out_path)

    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blend two expert checkpoints by splitting layers.",
    )
    parser.add_argument(
        "--expert1",
        default="experts/wiki.pt",
        help="Path to the first expert checkpoint (default: experts/wiki.pt).",
    )
    parser.add_argument(
        "--expert2",
        required=True,
        help="Path to the second expert checkpoint.",
    )
    parser.add_argument(
        "--split_layer",
        type=int,
        required=True,
        help="Layer index where expert1 ends and expert2 begins.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for the blended checkpoint.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    blend_experts(
        expert1_path=args.expert1,
        expert2_path=args.expert2,
        split_layer=args.split_layer,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()