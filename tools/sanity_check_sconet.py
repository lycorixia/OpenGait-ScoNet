#!/usr/bin/env python3
"""Smoke-test one Scoliosis1K silhouette PKL from each class through ScoNet.

This checks PKL loading, the configured silhouette transform, and a ScoNet
forward pass. Without --checkpoint, the randomly initialized logits are not
meaningful classification results.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENGAIT_DIR = REPO_ROOT / "opengait"
sys.path.insert(0, str(OPENGAIT_DIR))

from data.transform import BaseSilCuttingTransform
from modeling.models.sconet import ScoNet


LABELS = ("negative", "neutral", "positive")
LABEL_IDS = {name: index for index, name in enumerate(LABELS)}


def class_component(path: Path) -> str | None:
    for component in path.parts:
        normalized = component.casefold()
        if normalized in LABEL_IDS:
            return normalized
    return None


def find_one_pkl_per_class(dataset_root: Path) -> dict[str, Path]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    found: dict[str, Path] = {}
    for path in sorted(dataset_root.rglob("*.pkl")):
        label = class_component(path)
        if label is not None and label not in found:
            found[label] = path

    missing = [label for label in LABELS if label not in found]
    if missing:
        raise FileNotFoundError(
            f"No PKL found for {', '.join(missing)} under {dataset_root}"
        )
    return found


def load_frames(path: Path, max_frames: int) -> np.ndarray:
    with path.open("rb") as stream:
        data = pickle.load(stream)

    frames = np.asarray(data)
    if frames.ndim == 4:
        if frames.shape[1] == 1:
            frames = frames[:, 0]
        elif frames.shape[-1] == 1:
            frames = frames[..., 0]

    if frames.ndim != 3:
        raise ValueError(
            f"Expected PKL frames with shape [T,H,W], got {frames.shape} in {path}"
        )
    if frames.shape[0] == 0:
        raise ValueError(f"PKL contains no frames: {path}")

    if max_frames > 0 and len(frames) > max_frames:
        indexes = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = frames[indexes]

    transform = BaseSilCuttingTransform()
    processed = np.stack([transform(frame) for frame in frames]).astype(
        np.float32, copy=False
    )
    return np.ascontiguousarray(processed)


def build_sconet(config: dict, device: torch.device) -> ScoNet:
    # Build the repository's actual ScoNet modules without starting its
    # distributed training/evaluation engine.
    model = ScoNet.__new__(ScoNet)
    nn.Module.__init__(model)
    model.build_network(config["model_cfg"])
    model.init_parameters()
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model: ScoNet, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root containing identity/class/view/sequence PKL folders. "
             "Defaults to dataset_root in the config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "sconet" / "sconet_scoliosis1k.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "sanity_check_sconet.csv",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    dataset_root = args.dataset_root or Path(config["data_cfg"]["dataset_root"])
    samples = find_one_pkl_per_class(dataset_root)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA GPU.")
    device = (
        torch.device("cuda")
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else torch.device("cpu")
    )

    torch.manual_seed(0)
    np.random.seed(0)
    model = build_sconet(config, device)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: no checkpoint supplied; predictions use random weights.")

    rows = []
    for label in LABELS:
        pkl_path = samples[label]
        frames = load_frames(pkl_path, args.frames)
        sils = torch.from_numpy(frames).to(device=device).unsqueeze(0).unsqueeze(2)
        inputs = (
            [sils],
            torch.tensor([LABEL_IDS[label]], dtype=torch.long, device=device),
            [label],
            None,
            None,
        )
        with torch.inference_mode():
            result = model(inputs)
            logits = result["inference_feat"]["embeddings"].mean(dim=-1)
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        predicted = LABELS[int(probabilities.argmax())]
        row = {
            "true_label": label,
            "predicted_label": predicted,
            "confidence": float(probabilities.max()),
            "negative_probability": float(probabilities[0]),
            "neutral_probability": float(probabilities[1]),
            "positive_probability": float(probabilities[2]),
            "frames_used": len(frames),
            "input_height": int(frames.shape[1]),
            "input_width": int(frames.shape[2]),
            "pkl_path": str(pkl_path),
            "checkpoint": str(args.checkpoint) if args.checkpoint else "UNTRAINED",
        }
        rows.append(row)
        print(
            f"{label:8s} -> {predicted:8s} "
            f"(confidence={row['confidence']:.4f}, {len(frames)} frames)"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to: {args.output}")
    if args.checkpoint is None:
        print("These predictions only confirm that the data-to-model path runs; "
              "they are not an accuracy result.")


if __name__ == "__main__":
    main()
