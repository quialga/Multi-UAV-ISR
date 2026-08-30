"""
isr/tracking/metrics.py — MOT-Challenge style tracking metrics.

Written here rather than pulling in ``motmetrics`` so the thresholds stay
under our control and the project keeps its small dependency set.

Per frame, hypotheses are matched to ground truth by Hungarian on Euclidean
distance with a cut-off (``match_dist``).  From those matches:

  MOTP   mean position error over matched pairs      -> localisation quality
  IDSW   a GT object matched to a DIFFERENT hypothesis than last time
                                                     -> association failures
  MOTA   1 - (FN + FP + IDSW) / n_gt                 -> overall accuracy
  IDF1   identity-preserving F1 under the globally optimal GT<->hyp mapping
  MT/ML  GT trajectories tracked >=80% / <=20% of their life
  Frag   times a GT trajectory goes tracked -> lost -> tracked

Also carries filter-consistency checks, which MOT metrics do not cover:

  NEES   (x-x_hat)^T P^-1 (x-x_hat), should average the state dimension (4)
  NIS    y^T S^-1 y at update time, should average the measurement dimension

NEES/NIS are how you find out whether the filter is HONEST about its own
uncertainty.  Systematically high => Q too small (overconfident);
systematically low => Q too large.  That is exactly the failure the belief
map had (backlog §20: tight and wrong at long staleness), so it is worth
measuring rather than assuming.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from isr.tracking.assignment import solve_gated


class MOTAccumulator:
    """Accumulates per-frame matches and reports the summary metrics."""

    def __init__(self, match_dist: float = 5.0) -> None:
        self.match_dist = float(match_dist)
        self.n_gt = 0
        self.n_fp = 0
        self.n_fn = 0
        self.n_idsw = 0
        self.dists: List[float] = []
        self._last_match: Dict[int, int] = {}      # gt_id -> hyp_id
        # For IDF1 and MT/ML/Frag.
        self._co_occur: Dict[tuple, int] = defaultdict(int)   # (gt,hyp)->frames
        self._gt_frames: Dict[int, int] = defaultdict(int)
        self._hyp_frames: Dict[int, int] = defaultdict(int)
        self._gt_matched_frames: Dict[int, int] = defaultdict(int)
        self._gt_was_matched: Dict[int, bool] = {}
        self._frag: Dict[int, int] = defaultdict(int)

    def update(self, gt_ids: Sequence[int], gt_pos: np.ndarray,
               hyp_ids: Sequence[int], hyp_pos: np.ndarray) -> None:
        gt_ids = list(gt_ids)
        hyp_ids = list(hyp_ids)
        gt_pos = np.asarray(gt_pos, dtype=float).reshape(len(gt_ids), 2)
        hyp_pos = np.asarray(hyp_pos, dtype=float).reshape(len(hyp_ids), 2)

        self.n_gt += len(gt_ids)
        for g in gt_ids:
            self._gt_frames[g] += 1
        for h in hyp_ids:
            self._hyp_frames[h] += 1

        pairs = []
        if len(gt_ids) and len(hyp_ids):
            D = np.linalg.norm(gt_pos[:, None, :] - hyp_pos[None, :, :], axis=-1)
            gate = D <= self.match_dist
            pairs = solve_gated(D, gate)

        matched_gt, matched_hyp = set(), set()
        for gi, hi in pairs:
            g, h = gt_ids[gi], hyp_ids[hi]
            matched_gt.add(gi)
            matched_hyp.add(hi)
            self.dists.append(float(np.linalg.norm(gt_pos[gi] - hyp_pos[hi])))
            self._co_occur[(g, h)] += 1
            self._gt_matched_frames[g] += 1
            prev = self._last_match.get(g)
            if prev is not None and prev != h:
                self.n_idsw += 1
            self._last_match[g] = h
            if self._gt_was_matched.get(g) is False:
                self._frag[g] += 1          # lost -> tracked again
            self._gt_was_matched[g] = True

        self.n_fn += len(gt_ids) - len(matched_gt)
        self.n_fp += len(hyp_ids) - len(matched_hyp)
        for gi, g in enumerate(gt_ids):
            if gi not in matched_gt:
                self._gt_was_matched[g] = False

    # ------------------------------------------------------------------ #

    def _idf1(self) -> float:
        """IDF1 under the globally optimal one-to-one GT<->hypothesis map."""
        gts = sorted(self._gt_frames)
        hyps = sorted(self._hyp_frames)
        if not gts or not hyps:
            return 0.0
        # Maximise co-occurrence == minimise its negative.
        C = np.zeros((len(gts), len(hyps)))
        for i, g in enumerate(gts):
            for j, h in enumerate(hyps):
                C[i, j] = -self._co_occur.get((g, h), 0)
        pairs = solve_gated(C, C < 0)      # only pairs that ever co-occurred
        idtp = int(sum(-C[i, j] for i, j in pairs))
        idfn = sum(self._gt_frames.values()) - idtp
        idfp = sum(self._hyp_frames.values()) - idtp
        denom = 2 * idtp + idfp + idfn
        return float(2 * idtp / denom) if denom else 0.0

    def summary(self) -> Dict[str, float]:
        mota = (1.0 - (self.n_fn + self.n_fp + self.n_idsw) / self.n_gt) \
            if self.n_gt else float("nan")
        motp = float(np.mean(self.dists)) if self.dists else float("nan")
        ratios = [self._gt_matched_frames[g] / self._gt_frames[g]
                  for g in self._gt_frames]
        return {
            "MOTA": mota,
            "MOTP": motp,
            "IDF1": self._idf1(),
            "IDSW": float(self.n_idsw),
            "FP": float(self.n_fp),
            "FN": float(self.n_fn),
            "recall": float(1.0 - self.n_fn / self.n_gt) if self.n_gt else float("nan"),
            "MT": float(np.mean([r >= 0.8 for r in ratios])) if ratios else float("nan"),
            "ML": float(np.mean([r <= 0.2 for r in ratios])) if ratios else float("nan"),
            "Frag": float(sum(self._frag.values())),
            "n_gt": float(self.n_gt),
        }


class ConsistencyAccumulator:
    """NEES / NIS — is the filter honest about its own uncertainty?"""

    def __init__(self) -> None:
        self.nees: List[float] = []
        self.nis: List[float] = []

    def add_nees(self, x_hat: np.ndarray, P: np.ndarray,
                 x_true: np.ndarray) -> None:
        e = np.asarray(x_true, dtype=float) - np.asarray(x_hat, dtype=float)
        try:
            self.nees.append(float(e @ np.linalg.inv(P) @ e))
        except np.linalg.LinAlgError:
            pass

    def add_nis(self, values: Sequence[float]) -> None:
        self.nis.extend(float(v) for v in values)

    def summary(self, state_dim: int = 4) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.nees:
            out["NEES"] = float(np.mean(self.nees))
            out["NEES_target"] = float(state_dim)
        if self.nis:
            out["NIS"] = float(np.mean(self.nis))
        return out
