import numpy as np
from ..minmod_limiter import minmod, minmod_flag, switch_minmod


def test_minmod():
    a = np.array([-1, 1, 2, 4.])
    b = np.array([-3, 2, 0, 0])
    c = np.array([-5, 3, -1, 1.])
    np.testing.assert_allclose(minmod(a, b, c), np.array([-1, 1, 0, 0.]))


def test_minmod_flag():
    a = np.array([-1, 3, 2, -4, 5, 10])
    b = np.array([-3, 2, 1, -2, 0, -1])
    c = np.array([-5, 1, 1, -5.4, 1, 2])
    np.testing.assert_allclose(minmod_flag(a, b, c), np.array([1, 3, 2, 2, 0, 0]))


def test_switch_minmod():
    a = np.array([-1, 3, 2, -4, 5, 10])
    b = np.array([-3, 2, 1, -2, 0, -1])
    c = np.array([-5, 1, 1, -5.4, 1, 2])
    flag = minmod_flag(a, b, c)
    np.testing.assert_allclose(switch_minmod(a, b, c, flag), np.array([-1, 1, 1, -2, 0, 0.0]))
