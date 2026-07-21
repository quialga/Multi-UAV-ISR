"""
scripts/evaluate_trained.py — Stage 1 acceptance evaluation.

Given a trained checkpoint:

1. Evaluate (deterministic, N seeds) against each red baseline:
   stationary / random / run_from_nearest.
2. Re-run the heuristic baselines (Random / Greedy blue) on the same
   seeds for direct comparison.
3. Render one GIF per scenario (trained blue vs each red).
4. Also render Greedy blue vs Run on the same seed so the user can
   visually compare coordination behaviour.
5. Write a markdown results writeup to ``docs/stage1_results.md`` with:
   acceptance verdict, full eval table, GIF links, training metadata.

Run:
    python scripts/evaluate_trained.py --checkpoint runs/stage1/<run>/best.pt

Outputs land alongside the checkpoint:
    runs/stage1/<run>/eval_gifs/
    runs/stage1/<run>/eval_results.json
    docs/stage1_results.md  (overwritten each call — by design)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isr.agents.heuristics import (
    GreedyPursuer, HeuristicBlueAgent, RandomAgent,
    run_from_nearest_uav, stationary_red, random_red,
)
from isr.agents.policy_loader import (
    TrainedBlueAgent, build_trained_agent, load_policy,
)
from isr.env.pursuit_env import PursuitEnv
from isr.utils.render import animate_episode


# ---------------------------------------------------------------------------
# Episode rollout (works with any HeuristicBlueAgent)
# ---------------------------------------------------------------------------

def run_episode(
    blue:        HeuristicBlueAgent,
    red_policy:  Callable,
    env_kwargs:  Dict,
    seed:        int,
    collect_snaps: bool = False,
) -> Tuple[float, int, int, List]:
    """
    Roll out one episode.

    Returns
    -------
    total    : cumulative shared team reward
    n_caught : number of reds caught by episode end
    n_steps  : episode length in env steps (< max_steps if all reds
               caught early; == max_steps on timeout).  This is a much
               more coordination-sensitive metric than reward under
               the step-cost-dominated reward function.
    snaps    : list of state snapshots if collect_snaps else []
    """
    env = PursuitEnv(**env_kwargs, red_policy=red_policy, seed=seed)
    obs_d, _ = env.reset(seed=seed)
    total   = 0.0
    snaps: List[Dict] = []
    if collect_snaps:
        snaps.append(env.state_snapshot())
    while env.agents:
        actions = {a: blue.act(obs_d[a], env, a) for a in env.agents}
        obs_d, rew_d, term_d, trunc_d, info_d = env.step(actions)
        total += rew_d[env.possible_agents[0]]
        if collect_snaps:
            snaps.append(env.state_snapshot())
    snap = env.state_snapshot()
    n_caught = int((~snap["red_active"]).sum())
    n_steps  = int(snap["t"])
    return total, n_caught, n_steps, snaps


def eval_matrix(
    blue_factories: Dict[str, Callable],   # name -> ()->HeuristicBlueAgent
    red_factories:  Dict[str, Callable],   # name -> red_policy callable
    env_kwargs:     Dict,
    n_episodes:     int,
    seed_base:      int = 50_000,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns nested dict: results[blue_name][red_name] = {mean_return,
    std_return, mean_caught}.
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for b_name, b_factory in blue_factories.items():
        results[b_name] = {}
        for r_name, r_policy in red_factories.items():
            returns: List[float] = []
            caughts: List[int]   = []
            steps:   List[int]   = []
            for ep in range(n_episodes):
                blue = b_factory()
                # Same seed across blue policies => directly comparable
                seed = seed_base + ep
                r, c, n, _ = run_episode(blue, r_policy, env_kwargs, seed)
                returns.append(r)
                caughts.append(c)
                steps.append(n)
            results[b_name][r_name] = {
                "mean_return":  float(np.mean(returns)),
                "std_return":   float(np.std(returns)),
                "mean_caught":  float(np.mean(caughts)),
                "mean_steps":   float(np.mean(steps)),
                "std_steps":    float(np.std(steps)),
            }
    return results


# ---------------------------------------------------------------------------
# Markdown writeup
# ---------------------------------------------------------------------------

def _fmt_cell(d: Dict[str, float]) -> str:
    steps = d.get("mean_steps")
    step_str = f" — {steps:.1f} steps" if steps is not None else ""
    return (f"{d['mean_return']:+7.2f} ± {d['std_return']:.2f}  "
            f"({d['mean_caught']:.2f} caught{step_str})")


def write_results_md(
    out_path:        Path,
    ckpt_path:       Path,
    ckpt_meta:       Dict,
    results:         Dict[str, Dict[str, Dict[str, float]]],
    acceptance_bar:  float,
    trained_vs_run:  float,
    n_red:           int,
    gif_paths:       Dict[str, Path],
    eval_episodes:   int,
    stage_label:     str = "Stage 2",
    bar_label:       str = "1.20 × GreedyPursuer",
) -> None:
    """Render the results writeup.  ``stage_label`` + ``bar_label`` are
    populated by the caller so the same template serves Stage 1/2 and
    Stage 3 without a fork."""
    passed = trained_vs_run >= acceptance_bar
    verdict = "PASS" if passed else "FAIL"
    margin = trained_vs_run - acceptance_bar

    blue_names = list(results.keys())
    red_names  = list(next(iter(results.values())).keys())

    header = "| Blue \\\\ Red | " + " | ".join(red_names) + " |"
    align  = "|---" + "|---" * len(red_names) + "|"
    rows = []
    for b in blue_names:
        cells = [_fmt_cell(results[b][r]) for r in red_names]
        rows.append(f"| **{b}** | " + " | ".join(cells) + " |")
    table = "\n".join([header, align] + rows)

    gif_links = "\n".join(
        f"- [{name}]({path.as_posix()})" for name, path in gif_paths.items()
    )

    md = f"""# {stage_label} — Results

**Acceptance verdict: {verdict}**
(trained policy mean return vs `run_from_nearest_uav`: **{trained_vs_run:+.2f}**;
bar = {bar_label} = **{acceptance_bar:+.2f}**; margin = **{margin:+.2f}**)

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Checkpoint: `{ckpt_path}`
Training: rollout {ckpt_meta.get('rollout', '?')}, global_step {ckpt_meta.get('global_step', '?')}
Eval episodes per cell: {eval_episodes} (deterministic, fixed seeds shared across blue policies)

## Evaluation table

Rows: blue policy.  Columns: red policy.
Cell shows ``mean_return ± std_return  (mean caught of {n_red} — mean episode length in steps)`` — averaged
over {eval_episodes} episodes with matched seeds.

**Note on ``mean episode length``:** under the step-cost-dominated
reward (``-0.05`` per step), two policies with different coordination
quality can have very similar mean_return but noticeably different
episode lengths.  Fewer steps to catch all reds = more efficient
pursuit / better coordination.  Compare across the Trained row to see
whether a structured architecture is *behaviourally* more coordinated
even when reward moves only a little.

{table}

## Visualisations

{gif_links}

## Training arguments

```json
{json.dumps(ckpt_meta.get('args', {}), indent=2)}
```

## Interpretation notes

- The acceptance criterion (1.2× GreedyPursuer against the
  ``run_from_nearest_uav`` red) tests whether the learned policy does
  something a per-UAV nearest-target rule cannot — typically:
  splitting coverage across reds, anticipating where the red will
  dodge, or arriving from the direction red is fleeing.
- Reward components: +10 per catch, -0.01·||a||² (action cost), -0.05
  step cost, -5 terminal penalty per uncaught red.  A trained policy
  beating Greedy can do so by **catching the same number of reds
  faster** (less step cost), **catching with less effort** (lower
  action cost), or **catching more reds in time-up episodes** (less
  terminal penalty).
- Eval is deterministic (distribution mean, no exploration noise).
  Returns will be lower-variance than during training.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="path to a checkpoint .pt (best.pt or final.pt)")
    p.add_argument("--n-episodes", type=int, default=50,
                   help="episodes per (blue, red) cell")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--seed-base",  type=int, default=50_000,
                   help="shared seed base across blue policies for the same env")
    p.add_argument("--gif-seed",   type=int, default=12_345,
                   help="seed used for the qualitative GIFs (single episode)")
    p.add_argument("--fps",        type=int, default=10)
    p.add_argument("--results-md", default=None,
                   help="output markdown writeup path.  Defaults to "
                        "docs/stage2_results.md.")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt_path = args.checkpoint.resolve()
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # ---- Load policy + reconstruct env config from saved args -----------
    ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt_meta["args"]
    env_kwargs = dict(
        n_blue         = train_args["n_blue"],
        n_red          = train_args["n_red"],
        arena_size     = train_args["arena_size"],
        max_steps      = train_args["max_steps"],
        capture_radius = train_args["capture_radius"],
    )
    # Propagate sensor_radius if the checkpoint carries one, so the
    # trained policy sees the same observability regime it trained on.
    if train_args.get("sensor_radius") is not None:
        env_kwargs["sensor_radius"] = train_args["sensor_radius"]

    policy = load_policy(ckpt_path, device)
    policy_type = train_args.get("policy_type", "gnn")
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  policy_type={policy_type}")
    print(f"  rollout={ckpt_meta.get('rollout','?')}  "
          f"global_step={ckpt_meta.get('global_step','?')}")
    print(f"  env_kwargs={env_kwargs}")

    # ---- Eval matrix ----------------------------------------------------
    blue_factories = {
        "Random":  lambda: RandomAgent(seed=0),
        "Greedy":  lambda: GreedyPursuer(),
        # ``build_trained_agent`` picks the right adapter (Stage 1/2
        # TrainedBlueAgent) from the loaded policy's class.
        "Trained": lambda: build_trained_agent(policy, device, deterministic=True),
    }
    red_factories = {
        "Stationary":     stationary_red,
        "Random":         random_red(seed=0),
        "RunFromNearest": run_from_nearest_uav,
    }

    print(f"\nEvaluating ({args.n_episodes} episodes per cell, matched seeds)...")
    t0 = time.time()
    results = eval_matrix(
        blue_factories, red_factories, env_kwargs,
        n_episodes=args.n_episodes, seed_base=args.seed_base,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    # ---- Print the table to stdout -------------------------------------
    print("\nResults:")
    blue_names = list(results.keys())
    red_names  = list(next(iter(results.values())).keys())
    header = f"{'Blue':>10}  " + "  ".join(f"{r:>25}" for r in red_names)
    print(header)
    print("-" * len(header))
    for b in blue_names:
        cells = [_fmt_cell(results[b][r]) for r in red_names]
        print(f"{b:>10}  " + "  ".join(f"{c:>25}" for c in cells))

    # ---- Acceptance check ----------------------------------------------
    trained_vs_run = results["Trained"]["RunFromNearest"]["mean_return"]
    greedy_vs_run  = results["Greedy"]["RunFromNearest"]["mean_return"]
    trained_caught = results["Trained"]["RunFromNearest"]["mean_caught"]
    greedy_caught  = results["Greedy"]["RunFromNearest"]["mean_caught"]

    # Stage 1/2 acceptance = 1.2 × Greedy on mean_return.
    bar = 1.20 * greedy_vs_run
    verdict = "PASS" if trained_vs_run >= bar else "FAIL"
    print(f"\nAcceptance: trained={trained_vs_run:+.2f}  "
          f"bar=1.20×Greedy={bar:+.2f}  -> {verdict}")

    # ---- Render GIFs (one per red policy + one Greedy-vs-Run reference) -
    gif_dir = ckpt_path.parent / "eval_gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_paths: Dict[str, Path] = {}

    print(f"\nRendering GIFs to {gif_dir}/ ...")
    for r_name, r_policy in red_factories.items():
        # Trained blue vs this red — factory picks the right adapter.
        blue = build_trained_agent(policy, device, deterministic=True)
        _, _, _, snaps = run_episode(blue, r_policy, env_kwargs,
                                  seed=args.gif_seed, collect_snaps=True)
        path = gif_dir / f"trained_vs_{r_name.lower()}.gif"
        animate_episode(snaps, env_kwargs["arena_size"],
                        save_path=str(path), fps=args.fps)
        gif_paths[f"Trained vs {r_name}"] = path
        print(f"  {path.name}  ({len(snaps)} frames)")

    # Reference: Greedy blue vs Run, same seed as the trained-vs-Run GIF
    greedy = GreedyPursuer()
    _, _, _, snaps = run_episode(greedy, run_from_nearest_uav, env_kwargs,
                                 seed=args.gif_seed, collect_snaps=True)
    path = gif_dir / "greedy_vs_runfromnearest.gif"
    animate_episode(snaps, env_kwargs["arena_size"],
                    save_path=str(path), fps=args.fps)
    gif_paths["Greedy vs RunFromNearest (reference)"] = path
    print(f"  {path.name}  ({len(snaps)} frames)")

    # ---- Persist results.json + markdown writeup -----------------------
    results_json_path = ckpt_path.parent / "eval_results.json"
    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "checkpoint":   str(ckpt_path),
            "rollout":      ckpt_meta.get("rollout"),
            "global_step":  ckpt_meta.get("global_step"),
            "results":      results,
            "acceptance":   {
                "trained_vs_run": trained_vs_run,
                "bar":            bar,
                "verdict":        verdict,
            },
        }, f, indent=2)
    print(f"\nResults JSON: {results_json_path}")

    md_path = Path(args.results_md or "docs/stage2_results.md").resolve()
    stage_label = "Stage 2"
    bar_label = "1.20 × GreedyPursuer"
    write_results_md(
        out_path        = md_path,
        ckpt_path       = ckpt_path,
        ckpt_meta       = ckpt_meta,
        results         = results,
        acceptance_bar  = bar,
        trained_vs_run  = trained_vs_run,
        n_red           = env_kwargs["n_red"],
        gif_paths       = {k: v.relative_to(md_path.parent.parent)
                              if v.is_absolute() and md_path.parent.parent in v.parents
                              else v
                           for k, v in gif_paths.items()},
        eval_episodes   = args.n_episodes,
        stage_label     = stage_label,
        bar_label       = bar_label,
    )
    print(f"Markdown writeup: {md_path}")


if __name__ == "__main__":
    main()
