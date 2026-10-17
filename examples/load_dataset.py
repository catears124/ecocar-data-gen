"""Minimal consumer of a released corpus (no CARLA, no simulator).

    python examples/load_dataset.py path/to/ecocar-imitation-smoke-bicycle-v1.0.1 [--torch]

Loads train/val with the canonical extraction (policy input only), builds NumPy arrays, and fits
a ridge regression of the CERTIFIED longitudinal target as a smoke test of the pipeline. Steering
and pedal targets are available as ``y_steer``/``y_throttle``/``y_brake`` but carry no guarantee.
Samples of one run are correlated: split, group and evaluate by ``group`` (scenario), not by row."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autocross.datagen.dataset import load_examples, to_arrays  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--backend", default="bicycle", help="expected backend identity (bicycle or carla)")
    ap.add_argument("--torch", action="store_true", help="also wrap the arrays in a torch TensorDataset")
    a = ap.parse_args()
    tr = to_arrays(load_examples(a.corpus, "train", expect_backend=a.backend))
    va = to_arrays(load_examples(a.corpus, "val", expect_backend=a.backend))
    print(f"train {tr['x'].shape} from {len(set(tr['group']))} scenario groups; "
          f"val {va['x'].shape} from {len(set(va['group']))} groups; all certified: {bool(tr['certified'].all())}")
    mu, sd = tr["x"].mean(0), tr["x"].std(0) + 1e-9
    xt, xv = (tr["x"] - mu) / sd, (va["x"] - mu) / sd
    w = np.linalg.solve(xt.T @ xt + 1.0 * np.eye(xt.shape[1]), xt.T @ (tr["y_long"] - tr["y_long"].mean()))
    pred = xv @ w + tr["y_long"].mean()
    print(f"ridge baseline, certified longitudinal accel: val RMSE {np.sqrt(np.mean((pred - va['y_long']) ** 2)):.3f} m/s^2 "
          f"(target std {va['y_long'].std():.3f})")
    if a.torch:
        import torch
        ds = torch.utils.data.TensorDataset(torch.tensor(tr["x"], dtype=torch.float32),
                                            torch.tensor(tr["y_long"], dtype=torch.float32))
        print(f"torch TensorDataset with {len(ds)} rows")


if __name__ == "__main__":
    main()
