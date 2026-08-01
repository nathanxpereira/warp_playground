"""Tests for Cartpole/tuning/. Run from repo root: pytest tests/test_tuning.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Cartpole"))

from tuning.bryson import bryson_qr
from tuning.offline_sim import compute_gain, is_stable
from tuning.metrics import settling_time


def test_bryson_qr_diagonal_values():
    Q, R = bryson_qr(x_max=[0.5, 2.0, 0.1, 1.0], u_max=[20.0])
    np.testing.assert_allclose(np.diag(Q), [4.0, 0.25, 100.0, 1.0])
    np.testing.assert_allclose(R, [[0.0025]])


def test_is_stable_flags_zero_gain_as_unstable():
    # K=0 means no control -- an unactuated inverted pendulum is unstable.
    A = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 5, 0]])
    B = np.array([[0], [1], [0], [-1]])
    K = np.zeros(4)
    assert not is_stable(A, B, K)


def test_is_stable_true_for_valid_lqr_gain():
    A = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 5, 0]])
    B = np.array([[0], [1], [0], [-1]])
    Q, R = np.diag([4.0, 0.25, 100.0, 1.0]), np.array([[0.0025]])
    K = compute_gain(A, B, Q, R)
    assert is_stable(A, B, K)


def test_settling_time_inf_when_never_converges():
    t = np.linspace(0, 5, 300)
    angle = np.full_like(t, 0.5)  # never decays
    assert settling_time(t, angle, tol=0.05) == np.inf


def test_settling_time_finite_when_it_decays():
    t = np.linspace(0, 5, 300)
    angle = 0.5 * np.exp(-2 * t)
    st = settling_time(t, angle, tol=0.05)
    assert np.isfinite(st) and st > 0
