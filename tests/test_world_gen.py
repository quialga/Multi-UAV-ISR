"""
tests/test_world_gen.py — contract tests for the Gazebo world generator.

Pure-Python: parses the generated SDF as XML and checks it mirrors the
shadow env's sampled layout.  Does NOT need Gazebo installed (loading
the world in gz-sim is verified manually in WSL; see gazebo/README.md).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from gazebo.world_gen import env_to_sdf, make_env, ALT_BLUE, ALT_RED


@pytest.fixture(scope="module")
def env_and_root():
    env = make_env(seed=0, n_blue=5, n_red=3, n_obstacles=4,
                   arena_size=130.0)
    root = ET.fromstring(env_to_sdf(env))
    return env, root


def _models(root):
    return {m.get("name"): m for m in root.iter("model")}


def test_all_entities_present(env_and_root):
    env, root = env_and_root
    names = set(_models(root))
    assert {f"blue_{i}" for i in range(env.n_blue)} <= names
    assert {f"red_{j}" for j in range(env.n_red)} <= names
    assert {f"obstacle_{k}" for k in range(env.n_obstacles)} <= names
    assert {"wall_south", "wall_north", "wall_west", "wall_east",
            "ground_plane"} <= names


def test_spawn_poses_match_env(env_and_root):
    """Model poses must mirror the env's sampled spawn state exactly
    (identity coordinate mapping: env (x, y) -> Gazebo (x, y))."""
    env, root = env_and_root
    models = _models(root)
    for i in range(env.n_blue):
        x, y, z = map(float,
                      models[f"blue_{i}"].find("pose").text.split()[:3])
        assert abs(x - float(env._blue_pos[i, 0])) < 1e-3
        assert abs(y - float(env._blue_pos[i, 1])) < 1e-3
        assert z == ALT_BLUE
    for j in range(env.n_red):
        x, y, z = map(float,
                      models[f"red_{j}"].find("pose").text.split()[:3])
        assert abs(x - float(env._red_pos[j, 0])) < 1e-3
        assert abs(y - float(env._red_pos[j, 1])) < 1e-3
        assert z == ALT_RED


def test_obstacles_match_env_layout(env_and_root):
    env, root = env_and_root
    models = _models(root)
    for k in range(env.n_obstacles):
        m = models[f"obstacle_{k}"]
        assert m.find("static").text == "true"
        x, y = map(float, m.find("pose").text.split()[:2])
        assert abs(x - float(env._obstacle_pos[k, 0])) < 1e-3
        assert abs(y - float(env._obstacle_pos[k, 1])) < 1e-3
        r = float(m.find(".//cylinder/radius").text)
        assert abs(r - float(env._obstacle_r[k])) < 1e-3


def test_drones_are_kinematic_and_commandable(env_and_root):
    """Every drone needs (a) gravity off — it hovers without a
    controller — and (b) the VelocityControl plugin on its own
    /model/<name>/cmd_vel topic, which is the bridge's write interface."""
    env, root = env_and_root
    models = _models(root)
    drones = ([f"blue_{i}" for i in range(env.n_blue)]
              + [f"red_{j}" for j in range(env.n_red)])
    for name in drones:
        m = models[name]
        assert m.find(".//gravity").text == "false"
        plugins = {p.get("name"): p for p in m.findall("plugin")}
        assert "gz::sim::systems::VelocityControl" in plugins
        topic = plugins["gz::sim::systems::VelocityControl"].find("topic")
        assert topic.text == f"/model/{name}/cmd_vel"
        # Odometry out: per-drone topic so the bridge node knows WHO a
        # measurement belongs to (the fused pose topic loses names in
        # the ros_gz translation).
        assert "gz::sim::systems::OdometryPublisher" in plugins
        od = plugins["gz::sim::systems::OdometryPublisher"].find("odom_topic")
        assert od.text == f"/model/{name}/odometry"


def test_same_seed_same_world():
    a = env_to_sdf(make_env(0, 5, 3, 4, 130.0))
    b = env_to_sdf(make_env(0, 5, 3, 4, 130.0))
    assert a == b


def test_contact_surfaces_are_frictionless(env_and_root):
    """The env has no friction concept (wall-stop zeroes only the
    into-wall velocity component; sliding is free).  Gazebo's default
    friction wedged drones against walls permanently, so every surface
    a drone can touch must declare mu = 0."""
    env, root = env_and_root
    models = _models(root)
    touchable = ([f"blue_{i}" for i in range(env.n_blue)]
                 + [f"red_{j}" for j in range(env.n_red)]
                 + [f"obstacle_{k}" for k in range(env.n_obstacles)]
                 + ["wall_south", "wall_north", "wall_west", "wall_east"])
    for name in touchable:
        mu = models[name].find(".//collision//friction/ode/mu")
        assert mu is not None and float(mu.text) == 0.0, name
