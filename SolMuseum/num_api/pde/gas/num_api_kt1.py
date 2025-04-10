from numpy import sign, abs
from numba import njit


@njit(cache=True)
def kt1_ode0(pm1, p0, pp1, qm1, q0, qp1, S, va, lam, D, dx):
    return -1 / 2 * S * (-pm1 + pp1) / dx + (1 / 2) * va * (
            -2 * q0 + qm1 + qp1) / dx - 1 / 2 * lam * va ** 2 * q0 * abs(q0) / (D * S * p0)


@njit(cache=True)
def dkt1_ode0dq0(pm1, p0, pp1, qm1, q0, qp1, S, va, lam, D, dx):
    return -va / dx - 1 / 2 * lam * va ** 2 * q0 * sign(q0) / (D * S * p0) - 1 / 2 * lam * va ** 2 * abs(q0) / (
            D * S * p0)


@njit(cache=True)
def dkt1_ode0dqp1(pm1, p0, pp1, qm1, q0, qp1, S, va, lam, D, dx):
    return (1 / 2) * va / dx


@njit(cache=True)
def dkt1_ode0dpp1(pm1, p0, pp1, qm1, q0, qp1, S, va, lam, D, dx):
    return -1 / 2 * S / dx


@njit(cache=True)
def dkt1_ode0dqm1(pm1, p0, pp1, qm1, q0, qp1, S, va, lam, D, dx):
    return (1 / 2) * va / dx


@njit(cache=True)
def dkt1_ode0dpm1(pm1, p0, pp1, qm1, q0, qp1, S, va, lam, D, dx):
    return (1 / 2) * S / dx


@njit(cache=True)
def dkt1_ode0dp0(pm1, p0, pp1, qm1, q0, qp1, S, va, lam, D, dx):
    return (1 / 2) * lam * va ** 2 * q0 * abs(q0) / (D * S * p0 ** 2)
