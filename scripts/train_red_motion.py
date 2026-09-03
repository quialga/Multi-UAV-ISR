"""
scripts/train_red_motion.py — train the learned red-motion model.

Supervised training of ``RedMotionGNN`` on the collector's shards
(``scripts/collect_red_motion_dataset.py``), predicting a joint categorical
over the (heading, magnitude) grid plus the ZERO class.

Three decisions here are real; the rest is routine.

1. SOFT-LABEL cross-entropy, ``-sum_k q_k log p_k``.  The heading axis is
   CIRCULAR and bins are 10 degrees apart, so a hard label would call a
   neighbouring bin exactly as wrong as one pointing backwards.  ``q``
   comes from ``soft_targets`` — a fixed function of the ground-truth
   action, not something learned.  A hard label is the special case where
   ``q`` is one-hot, so this is the same cross-entropy, just with a less
   degenerate target.

2. SPLIT BY EPISODE, never by sample.  Consecutive steps of one episode are
   near-identical (the red moves ~1 m per step), so a random per-sample
   split would leak near-twins across train/val and report an optimistic
   validation number.

3. NO class reweighting, despite real imbalance (18.7% of samples sit at
   |a| = 1.0, the ZERO class is 1.78%).  These probabilities become the
   Gaussian-Sum BRANCH WEIGHTS at inference, so CALIBRATION matters more
   than balanced accuracy — reweighting would distort exactly the quantity
   the tracker consumes.

No input augmentation.  At inference the tracker samples hypotheses from
its own belief and asks this model "if the red were exactly HERE, what
would it do?" — the pointwise question exact-state training answers.
Augmenting the inputs and then marginalising would count the same
uncertainty twice.

Reported metrics go beyond loss, which is hard to read on its own:
  * ANGULAR ERROR in degrees — directly comparable to everything else
    measured for this adversary (its 0.315 rad/step median turn rate, the
    115-142 degree mode separations).
  * A MARGINAL BASELINE — what "predict the training marginal, ignore all
    inputs" scores.  With headings near-uniform in this data that floor
    sits around 90 degrees of angular error; a model that fails to beat it
    clearly has learned nothing.

Run:
    python scripts/train_red_motion.py --max-shards 4 --epochs 2   # smoke
    python scripts/train_red_motion.py --epochs 15
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isr.agents.red_motion_features import (
    N_BINS, N_HEADING_BINS, N_MAGNITUDE_BINS, ZERO_CLASS,
    featurize_shard, soft_targets,
)
from isr.agents.red_motion_gnn import RedMotionGNN

_INPUT_KEYS = ("red_feats", "blue_feats", "b2r_edge_feats",
              "obs_feats", "o2r_edge_feats", "b2r_active", "o2r_active")


# --------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------- #

def load_dataset(data_dir: str, max_shards: int = 0) -> Dict[str, torch.Tensor]:
    files = sorted(glob.glob(str(Path(data_dir) / "shard_*.npz")))
    if not files:
        raise SystemExit(f"no shards found in {data_dir}")
    if max_shards:
        files = files[:max_shards]
    parts: List[Dict[str, np.ndarray]] = []
    for f in files:
        parts.append(featurize_shard(dict(np.load(f))))
    out = {}
    for k in _INPUT_KEYS + ("target", "episode_id"):
        arr = np.concatenate([p[k] for p in parts], axis=0)
        out[k] = torch.from_numpy(arr)
    print(f"loaded {len(files)} shard(s): {out['target'].shape[0]} samples, "
          f"{sum(v.numel() * v.element_size() for v in out.values()) / 1e6:.0f} MB")
    return out


def split_by_episode(episode_id: torch.Tensor, val_frac: float, seed: int
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hold out whole EPISODES — see decision 2 in the module docstring."""
    eps = torch.unique(episode_id)
    g = torch.Generator().manual_seed(seed)
    perm = eps[torch.randperm(len(eps), generator=g)]
    n_val = max(1, int(round(val_frac * len(eps))))
    val_eps = set(perm[:n_val].tolist())
    is_val = torch.tensor([int(e) in val_eps for e in episode_id.tolist()])
    return (~is_val).nonzero(as_tuple=True)[0], is_val.nonzero(as_tuple=True)[0]


def soft_label_table() -> torch.Tensor:
    """(N_BINS, N_BINS): row k is the soft target for true class k.

    ``q`` depends ONLY on the class index, so there are just N_BINS
    distinct soft targets — precomputing all of them costs ~131 KB and
    turns per-batch label construction into an index_select.  Materialising
    them per sample instead would be 1.18M x 181 floats (856 MB).
    """
    return torch.from_numpy(soft_targets(np.arange(N_BINS)))


# --------------------------------------------------------------------- #
#  Loss and metrics
# --------------------------------------------------------------------- #

def soft_ce(logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return -(q * torch.log_softmax(logits, dim=-1)).sum(-1).mean()


def _heading_of(idx: torch.Tensor) -> torch.Tensor:
    """Bin centre heading in radians; NaN for the ZERO class (no heading)."""
    h = (idx // N_MAGNITUDE_BINS).to(torch.float64)
    ang = (h + 0.5) / N_HEADING_BINS * 2 * np.pi - np.pi
    return torch.where(idx == ZERO_CLASS, torch.full_like(ang, float("nan")), ang)


def angular_error_deg(pred_idx: torch.Tensor, true_idx: torch.Tensor
                      ) -> torch.Tensor:
    """Circular |difference| in degrees over samples whose TRUE class has a
    heading (the ZERO class does not)."""
    a, b = _heading_of(pred_idx), _heading_of(true_idx)
    ok = ~torch.isnan(b)
    d = a[ok] - b[ok]
    d = torch.where(torch.isnan(d), torch.full_like(d, np.pi), d)  # pred ZERO
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return d.abs() * 180.0 / np.pi


@torch.no_grad()
def evaluate(model, data, idx, table, batch: int) -> Dict[str, float]:
    model.eval()
    tot_loss, n = 0.0, 0
    preds, trues = [], []
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        logits = model(*[data[k][b] for k in _INPUT_KEYS])[:, 0]
        t = data["target"][b]
        tot_loss += float(soft_ce(logits, table[t])) * len(b)
        n += len(b)
        preds.append(logits.argmax(-1))
        trues.append(t)
    pred, true = torch.cat(preds), torch.cat(trues)
    ang = angular_error_deg(pred, true)
    return {
        "loss": tot_loss / n,
        "acc": float((pred == true).float().mean()),
        "ang_med": float(ang.median()),
        "ang_mean": float(ang.mean()),
        "zero_recall": (float((pred[true == ZERO_CLASS] == ZERO_CLASS)
                             .float().mean())
                       if int((true == ZERO_CLASS).sum()) else float("nan")),
    }


def marginal_baseline(train_t: torch.Tensor, val_t: torch.Tensor,
                      table: torch.Tensor) -> Dict[str, float]:
    """The honest floor: predict the TRAIN marginal, ignore every input.

    Loss and accuracy use the full marginal.  The ANGULAR floor uses the
    best fixed HEADING instead, deliberately: the marginal's argmax turns
    out to be the ZERO class (1.57%, more common than any single grid cell
    once headings spread near-uniformly), which has no heading at all and
    would score the maximal 180 degrees — flattering the model against a
    straw floor.  The honest "no directional information" baseline is the
    single best heading, which against near-uniform truth sits at ~90 deg.
    """
    counts = torch.bincount(train_t, minlength=N_BINS).double()
    p = (counts / counts.sum()).clamp_min(1e-12)
    logits = p.log().float()
    loss = float(soft_ce(logits.expand(len(val_t), -1), table[val_t]))
    pred = torch.full_like(val_t, int(p.argmax()))
    best_grid = int(counts[:ZERO_CLASS].argmax())      # best class WITH a heading
    ang = angular_error_deg(torch.full_like(val_t, best_grid), val_t)
    return {"loss": loss, "acc": float((pred == val_t).float().mean()),
            "ang_med": float(ang.median()), "ang_mean": float(ang.mean()),
            "zero_recall": float("nan")}


# --------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=str, default="data/red_motion")
    p.add_argument("--max-shards", type=int, default=0, help="0 = all")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--d-hidden", type=int, default=64)
    p.add_argument("--msg-rounds", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/red_motion/model.pt")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    data = load_dataset(a.data, a.max_shards)
    tr_idx, va_idx = split_by_episode(data["episode_id"], a.val_frac, a.seed)
    n_ep = len(torch.unique(data["episode_id"]))
    print(f"split by EPISODE: {len(tr_idx)} train / {len(va_idx)} val samples "
          f"from {n_ep} episodes")

    table = soft_label_table()
    base = marginal_baseline(data["target"][tr_idx], data["target"][va_idx],
                            table)
    print(f"\nmarginal baseline (ignores all inputs): loss {base['loss']:.4f}  "
          f"acc {base['acc']:.3f}  angular median {base['ang_med']:.1f} deg\n")

    blue_cap = data["blue_feats"].shape[1]
    obs_cap = data["obs_feats"].shape[1]
    model = RedMotionGNN(n_blue=blue_cap, n_red=1, n_obs=obs_cap,
                        d_hidden=a.d_hidden, n_msg_rounds=a.msg_rounds)
    print(f"model: {sum(q.numel() for q in model.parameters())} params "
          f"(n_blue={blue_cap}, n_obs={obs_cap}, {N_BINS} classes)")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    best = float("inf")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        run, nb = 0.0, 0
        for s in range(0, len(perm), a.batch_size):
            b = perm[s:s + a.batch_size]
            logits = model(*[data[k][b] for k in _INPUT_KEYS])[:, 0]
            loss = soft_ce(logits, table[data["target"][b]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1
        m = evaluate(model, data, va_idx, table, a.batch_size)
        flag = ""
        if m["loss"] < best:
            best = m["loss"]
            torch.save({"state_dict": model.state_dict(),
                       "n_blue": blue_cap, "n_obs": obs_cap,
                       "d_hidden": a.d_hidden, "msg_rounds": a.msg_rounds},
                      a.out)
            flag = "  *saved"
        print(f"[{ep:3d}/{a.epochs}] train {run / nb:.4f} | val {m['loss']:.4f} "
              f"acc {m['acc']:.3f} ang_med {m['ang_med']:5.1f} deg "
              f"zero_rec {m['zero_recall']:.2f} | {time.time() - t0:.0f}s{flag}")

    print(f"\nbest val loss {best:.4f} -> {a.out}")
    print(f"vs marginal baseline {base['loss']:.4f} "
          f"(angular {base['ang_med']:.1f} deg)")


if __name__ == "__main__":
    main()
