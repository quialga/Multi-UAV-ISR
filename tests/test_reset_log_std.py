"""
tests/test_reset_log_std.py — the exploration width a warm start hands to
PPO.

The trap this exists for.  PPO **samples** its rollout actions; the
``[det eval]`` metric takes the distribution mean.  Behaviour cloning
regresses only the mean and deliberately leaves ``log_std`` alone (the
MSE has no gradient into it), so a cloned actor arrives with log_std at
its init 0.0 — sigma 1.0, which on a ``[-1, 1]^2`` action box is close to
uniform noise.  Measured on ``bc_tracker_v1``: **2.52/3 deterministic
against 1.70/3 sampled**, and 2.20 vs 0.75 on the evader.  Fine-tuning
from there would compute advantages on trajectories that do not reflect
the policy, and read afterwards as "PPO destroyed the clone".

Why it has to run AFTER the warm start, which is the whole point of the
flag: ``load_full_stage4`` copies every shape-matching tensor, and
``actor_log_std`` matches, so the checkpoint's own sigma silently
overwrites whatever the constructor was given.

Run:
    pytest tests/test_reset_log_std.py -v
"""
from __future__ import annotations

import math

import pytest
import torch

from isr.agents.gnn_stage4_policy import GNNStage4Policy

KW = dict(n_blue=3, n_red=2, n_obs=0)


@pytest.fixture
def donor_ckpt(tmp_path):
    """A checkpoint carrying the width behaviour cloning leaves behind:
    log_std at its init 0.0, i.e. sigma 1.0."""
    donor = GNNStage4Policy(**KW, init_log_std=0.0)
    p = tmp_path / "donor.pt"
    torch.save({"policy_state": donor.state_dict()}, p)
    return donor, p


def test_the_warm_start_copy_overwrites_the_constructors_log_std(donor_ckpt):
    """The reason a constructor argument is not enough, and therefore the
    reason the flag applies after the copy.  If this ever stops being
    true, --reset-log-std can move earlier."""
    _, path = donor_ckpt
    target = GNNStage4Policy(**KW, init_log_std=-1.2)
    assert torch.allclose(target.actor_log_std, torch.full((2,), -1.2))

    target.load_full_stage4(path)
    assert torch.allclose(target.actor_log_std, torch.zeros(2)), (
        "the checkpoint's log_std should have overwritten the "
        "constructor's — if not, the flag's rationale has changed")


def test_resetting_after_the_copy_sticks(donor_ckpt):
    """What --reset-log-std does, in the order it does it."""
    _, path = donor_ckpt
    target = GNNStage4Policy(**KW, init_log_std=0.0)
    target.load_full_stage4(path)
    with torch.no_grad():
        target.actor_log_std.fill_(-1.2)

    assert torch.allclose(target.actor_log_std, torch.full((2,), -1.2))
    sigma = float(target.actor_log_std.detach().exp()[0])
    assert abs(sigma - math.exp(-1.2)) < 1e-6
    assert 0.25 < sigma < 0.35, f"sigma {sigma} outside the intended range"


def test_reset_keeps_log_std_trainable():
    """It is a starting width, not a freeze: PPO's entropy term still has
    to be able to move it."""
    policy = GNNStage4Policy(**KW, init_log_std=0.0)
    with torch.no_grad():
        policy.actor_log_std.fill_(-1.2)
    assert policy.actor_log_std.requires_grad
    loss = policy.actor_log_std.sum()
    loss.backward()
    assert policy.actor_log_std.grad is not None
    assert torch.allclose(policy.actor_log_std.grad, torch.ones(2))


def test_reset_touches_nothing_but_log_std(donor_ckpt):
    """A warm start is worth having; the flag must not disturb it."""
    donor, path = donor_ckpt
    target = GNNStage4Policy(**KW, init_log_std=0.0)
    target.load_full_stage4(path)
    with torch.no_grad():
        target.actor_log_std.fill_(-1.2)

    for k, v in donor.state_dict().items():
        if k == "actor_log_std":
            continue
        assert torch.equal(target.state_dict()[k], v), k
