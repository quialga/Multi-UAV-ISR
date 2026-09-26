"""The red-1.5 reference table, with standard errors.

Everything downstream (comms_radius above all) gets measured against this,
so the numbers cannot come from the training log: those are 25 episodes with
no error bar, and Sec. 13.5's lesson was that quoting a single eval quotes
the peak of the noise.

50 matched seeds per cell.  Rows, and what each one isolates:

  ObsGreedy / AssignGreedy   the heuristic floor, in the SAME partial-obs
                             env as the policy.  The training log's Greedy
                             baseline runs with sensor_radius=None, i.e.
                             full observability, so it is not comparable.
  red14/best  AT 1.5         the 1.4 policy transplanted with NO retraining
                             -- isolates what the curriculum step bought.
  red15_v1/final             first cycle, lr 3e-05.
  red15_v2/{best,final}      second cycle, lr 6e-05.

The last block re-runs the winner at red 1.4 to check the curriculum did not
cost anything on the easier regime it came from.
"""
import sys
sys.path.insert(0, r"C:\Users\quial\sources\Multi-UAV-ISR")
import numpy as np, torch
from isr.agents.heuristics import (run_from_nearest_uav,
                                   ObservationGreedyPursuer,
                                   AssignmentGreedyPursuer)
from isr.agents.policy_loader import (env_kwargs_from_checkpoint, load_policy,
                                      build_trained_agent)
from isr.env.pursuit_env import PursuitEnv

DEV = torch.device("cpu")
N_EP = 50
REF = r"runs/stage4/ppo_red15_v2/final.pt"
base = env_kwargs_from_checkpoint(
    torch.load(REF, map_location="cpu", weights_only=False)["args"])


def evaluate(make_blue, v_max, n_ep=N_EP):
    caught, steps, rets = [], [], []
    kw = dict(base, red_v_max=v_max)
    for ep in range(n_ep):
        seed = 70_000 + ep
        env = PursuitEnv(**kw, red_policy=run_from_nearest_uav, seed=seed)
        env.reset(seed=seed)
        blue = make_blue()
        tot = 0.0
        while env.agents:
            acts = {a: blue.act(None, env, a) for a in env.agents}
            _o, rew, _t, _tr, _i = env.step(acts)
            tot += float(sum(rew.values())) / max(len(rew), 1)
        s = env.state_snapshot()
        caught.append(int((~s["red_active"]).sum()))
        steps.append(int(s["t"]))
        rets.append(tot)
    c = np.array(caught, dtype=float)
    return (c.mean(), c.std(ddof=1) / np.sqrt(len(c)),
            np.mean(steps), np.mean(rets))


def policy_factory(path):
    pol = load_policy(path, DEV)
    return lambda: build_trained_agent(pol, DEV, deterministic=True)


ROWS = [
    ("ObsGreedy",            lambda: ObservationGreedyPursuer(),      1.5),
    ("AssignGreedy",         lambda: AssignmentGreedyPursuer(),       1.5),
    ("red14/best @1.5",      policy_factory("runs/stage4/ppo_red14_v1/best.pt"), 1.5),
    ("red15_v1/final",       policy_factory("runs/stage4/ppo_red15_v1/final.pt"), 1.5),
    ("red15_v2/best",        policy_factory("runs/stage4/ppo_red15_v2/best.pt"), 1.5),
    ("red15_v2/final",       policy_factory("runs/stage4/ppo_red15_v2/final.pt"), 1.5),
    ("red15_v2/final @1.4",  policy_factory("runs/stage4/ppo_red15_v2/final.pt"), 1.4),
    ("red14/best @1.4",      policy_factory("runs/stage4/ppo_red14_v1/best.pt"), 1.4),
]

print(f"{N_EP} matched seeds, red = run_from_nearest_uav, deterministic, "
      f"partial obs (tracker)\n")
print(f"{'blue':>22}  {'red':>5}  {'caught':>14}  {'steps':>7}  {'return':>8}")
print("-" * 66)
for name, factory, v in ROWS:
    m, se, st, r = evaluate(factory, v)
    print(f"{name:>22}  {v:5.2f}  {m:6.2f} +- {se:4.2f}  {st:7.1f}  "
          f"{r:8.2f}", flush=True)
