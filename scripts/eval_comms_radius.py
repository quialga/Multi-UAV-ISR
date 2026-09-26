"""Free eval-time ablation of comms_radius on the red-1.5 reference.

comms_radius changes no tensor shapes -- bb edges are already complete,
n_blue*(n_blue-1); only the bb_edge_visible MASK moves -- so the existing
checkpoint loads and runs under any value without retraining.  That makes
this ablation cost nothing, which is the whole reason to run it before
spending ~5.5 h on a retrain.

READ IT ASYMMETRICALLY.  The policy was TRAINED with comms gated at
sensor_radius 40, so opening the radio feeds it inputs it has never seen:
ally edges that were always zero beyond 40 m suddenly carry messages.  If
the score goes UP anyway that is strong evidence -- the information is
useful even unlearned.  If it goes DOWN it proves nothing, because
out-of-distribution inputs degrading a policy is the expected null result,
not evidence that unlimited comms is bad.

Sweeps down as well as up: narrowing the radio below 40 should HURT if the
ally channel matters at all, and a flat response to narrowing would say the
policy barely uses ally edges -- which would itself predict that widening
them does nothing.
"""
import sys
sys.path.insert(0, r"C:\Users\quial\sources\Multi-UAV-ISR")
import numpy as np, torch
from isr.agents.heuristics import run_from_nearest_uav
from isr.agents.policy_loader import (env_kwargs_from_checkpoint, load_policy,
                                      build_trained_agent)
from isr.env.pursuit_env import PursuitEnv

CK = r"runs/stage4/ppo_red15_v2/final.pt"
DEV = torch.device("cpu")
N_EP = 50
base = env_kwargs_from_checkpoint(
    torch.load(CK, map_location="cpu", weights_only=False)["args"])
policy = load_policy(CK, DEV)
print(f"checkpoint {CK}")
print(f"trained with sensor_radius={base['sensor_radius']}, "
      f"comms_radius={base.get('comms_radius')}  (None = gated at sensor)")
print(f"\n{N_EP} matched seeds, red v_max 1.5, run_from_nearest_uav, "
      f"deterministic\n")
print(f"{'comms_radius':>14}  {'caught':>14}  {'steps':>7}  {'bb open %':>9}")
print("-" * 52)

for cr in (20.0, 40.0, 60.0, 90.0, 130.0, float("inf")):
    caught, steps, bb_frac = [], [], []
    for ep in range(N_EP):
        seed = 70_000 + ep
        env = PursuitEnv(**dict(base, red_v_max=1.5, comms_radius=cr),
                         red_policy=run_from_nearest_uav, seed=seed)
        env.reset(seed=seed)
        blue = build_trained_agent(policy, DEV, deterministic=True)
        while env.agents:
            bb, _rb = env._compute_edge_visibility()
            bb_frac.append(float(bb.mean()))
            env.step({a: blue.act(None, env, a) for a in env.agents})
        s = env.state_snapshot()
        caught.append(int((~s["red_active"]).sum()))
        steps.append(int(s["t"]))
    c = np.array(caught, dtype=float)
    se = c.std(ddof=1) / np.sqrt(len(c))
    label = "inf" if np.isinf(cr) else f"{cr:.0f}"
    mark = "  <-- as trained" if cr == 40.0 else ""
    print(f"{label:>14}  {c.mean():6.2f} +- {se:4.2f}  {np.mean(steps):7.1f}  "
          f"{100*np.mean(bb_frac):8.1f}%{mark}", flush=True)
