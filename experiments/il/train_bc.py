"""Train a small deterministic behaviour-cloning network on an IL corpus.

The corpus loader is intentionally kept at the boundary of this module.  The
network itself only consumes the numerical arrays returned by ``experiments.il
.dataset`` and therefore does not depend on corpus record details at training
time.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

try:  # Torch is an optional dependency of the repository.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in minimal installs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:

    class BCNet(nn.Module):
        """Two-layer ReLU trunk with independent offset and acceleration heads."""

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.offset = nn.Linear(128, 1)
            self.accel = nn.Linear(128, 1)

        def forward(self, x):
            h = self.trunk(x)
            return torch.cat((self.offset(h), self.accel(h)), dim=1)

else:

    class BCNet:  # type: ignore[no-redef]
        """Placeholder keeping this module importable without torch."""

        def __init__(self, input_dim: int) -> None:
            raise RuntimeError("train_bc requires the optional torch dependency")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("train_bc requires the optional torch dependency")
    return torch


def set_seed(seed: int) -> None:
    """Seed every RNG used by this CPU training path."""
    random.seed(seed)
    np.random.seed(seed)
    t = _require_torch()
    t.manual_seed(seed)
    # The model is deliberately CPU-only.  Single-threaded kernels avoid
    # reduction-order changes between runs on otherwise identical machines.
    t.set_num_threads(1)
    try:
        t.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):  # older torch builds
        pass


def _records(corpus: str | Path, condition: str, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one split while accepting both generations of the dataset API."""
    from . import dataset

    loaded = dataset.load_examples(corpus, condition=condition, split=split, return_stats=True)
    if isinstance(loaded, tuple) and len(loaded) == 2:
        rows, stats = loaded
        return list(rows), dict(stats)
    return list(loaded), {}


def _arrays(rows: list[dict[str, Any]], dropped: int = 0):
    from .dataset import to_arrays

    return to_arrays(rows, dropped_infeasible=dropped)


def _take_budget(rows: list[dict[str, Any]], budget: int, seed: int) -> list[dict[str, Any]]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if len(rows) <= budget:
        return rows
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(rows), size=budget, replace=False))
    return [rows[int(i)] for i in chosen]


def _state_numpy(model: BCNet) -> dict[str, np.ndarray]:
    """Export a stable, controller-friendly parameter layout."""
    state = model.state_dict()
    return {
        "W0": state["trunk.0.weight"].detach().cpu().numpy().T.astype(np.float32),
        "b0": state["trunk.0.bias"].detach().cpu().numpy().astype(np.float32),
        "W1": state["trunk.2.weight"].detach().cpu().numpy().T.astype(np.float32),
        "b1": state["trunk.2.bias"].detach().cpu().numpy().astype(np.float32),
        "W_offset": state["offset.weight"].detach().cpu().numpy().T.astype(np.float32),
        "b_offset": state["offset.bias"].detach().cpu().numpy().astype(np.float32),
        "W_accel": state["accel.weight"].detach().cpu().numpy().T.astype(np.float32),
        "b_accel": state["accel.bias"].detach().cpu().numpy().astype(np.float32),
    }


def train(
    corpus: str | Path,
    condition: str,
    budget: int,
    seed: int,
    out: str | Path,
    *,
    max_epochs: int = 1000,
    patience: int = 40,
) -> dict[str, Any]:
    """Fit the BC model and write its complete inference artifact."""
    t = _require_torch()
    set_seed(seed)

    train_rows, train_stats = _records(corpus, condition, "train")
    val_rows, val_stats = _records(corpus, condition, "val")
    train_rows = _take_budget(train_rows, budget, seed)
    train_data = _arrays(train_rows, int(train_stats.get("dropped_infeasible", 0)))
    val_data = _arrays(val_rows, int(val_stats.get("dropped_infeasible", 0)))
    if train_data.x.ndim != 2 or train_data.x.shape[0] == 0:
        raise ValueError("selected corpus has no feasible training examples")
    if val_data.x.ndim != 2 or val_data.x.shape[0] == 0:
        raise ValueError("selected corpus has no feasible validation examples")
    if train_data.x.shape[1] != val_data.x.shape[1]:
        raise ValueError("train and validation feature dimensions differ")

    x_train = np.asarray(train_data.x, dtype=np.float32)
    x_val = np.asarray(val_data.x, dtype=np.float32)
    y_train = np.asarray(train_data.y, dtype=np.float32)
    y_val = np.asarray(val_data.y, dtype=np.float32)
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std < 1.0e-8)] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    # Copies are intentionally made on CPU: this artifact is consumed by the
    # lightweight numpy controller and must not require CUDA at inference.
    xt = t.as_tensor(x_train, dtype=t.float32, device="cpu")
    yt = t.as_tensor(y_train, dtype=t.float32, device="cpu")
    xv = t.as_tensor(x_val, dtype=t.float32, device="cpu")
    yv = t.as_tensor(y_val, dtype=t.float32, device="cpu")
    model = BCNet(int(xt.shape[1])).to("cpu")
    optimizer = t.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xt), yt)
        loss.backward()
        optimizer.step()

        model.eval()
        with t.no_grad():
            val_mae = float(t.mean(t.abs(model(xv) - yv)).item())
        if val_mae < best_mae - 1.0e-8:
            best_mae = val_mae
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:  # max_epochs is user-adjustable; retain a useful error.
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)

    metadata = {
        "format": "veridrive-il-bc/1",
        "condition": condition,
        "budget": int(budget),
        "seed": int(seed),
        "train_count": int(len(train_rows)),
        "val_count": int(len(val_rows)),
        "input_dim": int(x_train.shape[1]),
        "hidden_width": 128,
        "hidden_layers": 2,
        "activation": "relu",
        "heads": ["offset", "accel"],
        "target_units": ["m", "m/s^2"],
        "best_epoch": int(best_epoch),
        "epochs": int(epoch),
        "val_mae": float(best_mae),
        "dropped_infeasible_train": int(train_stats.get("dropped_infeasible", 0)),
        "dropped_infeasible_val": int(val_stats.get("dropped_infeasible", 0)),
    }
    artifact = _state_numpy(model)
    artifact.update({
        "x_mean": mean,
        "x_std": std,
        # Short aliases make the normalization contract convenient for small
        # numpy consumers while retaining explicit names for new consumers.
        "mean": mean,
        "std": std,
        "grid": np.asarray(train_data.grid, dtype=np.float32),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    })
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **artifact)
    print(f"TRAIN_OK val_mae={best_mae:.8g} out={output}")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a deterministic CPU behaviour-cloning MLP")
    parser.add_argument("--corpus", required=True, help="IL corpus directory")
    parser.add_argument("--condition", required=True, choices=("optimal", "approx", "heuristic", "cert_control"))
    parser.add_argument("--budget", required=True, type=int, help="maximum number of training examples")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, help="output .npz artifact path")
    args = parser.parse_args(argv)
    train(args.corpus, args.condition, args.budget, args.seed, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
