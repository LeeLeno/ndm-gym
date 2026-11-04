"""Structured semi-implicit SuperMimic increment world model."""

import pytest
import torch

from conftest import random_states


@pytest.fixture(scope="module")
def batch(motion):
    from ndm_gym.state import normalize_xi

    f, S, xi = random_states(motion, 32, seed=11)
    g = torch.Generator().manual_seed(12)
    p, R, v, w, q, qd = S
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    S1 = (p + 0.02 * v + 0.001 * rn(32, 3), R, v + 0.2 * rn(32, 3), w + 0.2 * rn(32, 3), q + 0.01 * rn(32, 29), qd + 0.3 * rn(32, 29))
    a = rn(32, 29)
    return S, S1, a, normalize_xi(xi)


def test_structured_targets_integrate_back_to_next_state(batch):
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    S, S1, _, _ = batch
    wm = IncrementWorldModel(structured=True)
    target = wm.regression_targets(S, S1)
    assert target.shape == (32, 70)
    assert torch.equal(target, wm.targets(S, S1)) is False
    predicted = wm.integrate(S, target)
    torch.testing.assert_close(predicted, S1, rtol=1e-5, atol=1e-6)


def test_zero_residual_is_semi_implicit_drift(batch):
    from ndm_gym.core.rotations import so3_exp, yaw_mat
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    S, _, _, _ = batch
    p, R, v, w, q, qd = S
    g = torch.Generator().manual_seed(13)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    v1, w1, qd1 = v + rn(*v.shape), w + rn(*w.shape), qd + rn(*qd.shape)
    dt = IncrementWorldModel.DRIFT_NEW_VELOCITY_WEIGHT
    old = 1.0 - dt
    dt *= IncrementWorldModel().dt
    old *= IncrementWorldModel().dt
    H = yaw_mat(R)
    pos_drift = (H.transpose(-1, -2) @ (old * v + dt * v1).unsqueeze(-1)).squeeze(-1)
    p1 = p + (H @ pos_drift.unsqueeze(-1)).squeeze(-1)
    rot_drift = (R.transpose(-1, -2) @ (old * w + dt * w1).unsqueeze(-1)).squeeze(-1)
    R1 = R @ so3_exp(rot_drift)
    q1 = q + old * qd + dt * qd1
    S1 = (p1, R1, v1, w1, q1, qd1)

    wm = IncrementWorldModel(structured=True)
    target = wm.regression_targets(S, S1)
    pos, rot, lin, ang, joint, joint_vel = target.split((3, 3, 3, 3, 29, 29), -1)
    torch.testing.assert_close(pos, torch.zeros_like(pos), atol=1e-6, rtol=0)
    torch.testing.assert_close(rot, torch.zeros_like(rot), atol=1e-6, rtol=0)
    torch.testing.assert_close(joint, torch.zeros_like(joint), atol=1e-6, rtol=0)
    torch.testing.assert_close(lin, (R.transpose(-1, -2) @ (v1 - v).unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(ang, (R.transpose(-1, -2) @ (w1 - w).unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(joint_vel, qd1 - qd)
    target[..., :6] = 0
    target[..., 12:15] = 0
    torch.testing.assert_close(wm.integrate(S, target), S1, rtol=1e-5, atol=1e-6)


def test_increment_error_norm_stays_physical_and_old_checkpoints_load(batch):
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    S, S1, _, _ = batch
    old = IncrementWorldModel(hidden=(16, 16))
    old.in_norm.update(old.inputs(S, torch.randn(32, 29), torch.randn(32, 33)))
    old.out_norm.update(old.targets(S, S1))
    state = {k: v.clone() for k, v in old.state_dict().items() if not k.startswith("inc_norm.")}

    unstructured = IncrementWorldModel(hidden=(16, 16))
    unstructured.load_state_dict(state)
    torch.testing.assert_close(unstructured.inc_norm.mean, old.out_norm.mean)
    torch.testing.assert_close(unstructured.inc_norm.var, old.out_norm.var)

    structured = IncrementWorldModel(hidden=(16, 16), structured=True)
    with pytest.raises(RuntimeError, match="inc_norm"):
        structured.load_state_dict(state)


def test_rigidformer_backbone_supports_structured_output(kin, batch):
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    S, S1, a, xi_n = batch
    rf = dict(kin=kin, dim=32, layers=1, heads=2, registers=2)
    wm = IncrementWorldModel(rigidformer=rf, structured=True)
    wm.in_norm.update(wm.inputs(S, a, xi_n))
    target = wm.regression_targets(S, S1)
    wm.out_norm.update(target)
    wm.inc_norm.update(wm.targets(S, S1))
    with torch.no_grad():
        wm.net.root_head.weight.normal_(0, 0.1)
        wm.net.root_head.bias.normal_(0, 0.1)
        wm.net.joint_head.weight.normal_(0, 0.1)
        wm.net.joint_head.bias.normal_(0, 0.1)
    action = a.clone().requires_grad_()
    out = wm.step(S, action, xi_n)
    loss = sum(x.square().mean() for x in out)
    loss.backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    assert action.grad.abs().sum() > 0


def test_world_model_args_record_structure():
    from ndm_gym.train import add_train_args

    import argparse

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args(["--algo", "supermimic", "--wm_backbone", "rigidformer", "--wm_structured"])
    assert args.wm_structured is True
