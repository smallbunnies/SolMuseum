from Solverz import Eqn, Ode, AliasVar, TimeSeriesParam, Param
from Solverz import iVar, idx, Var, Abs
from Solverz.utilities.type_checker import is_integer, is_number
from sympy import Add, Mul, Rational, Integer
import numpy as np

from .weno3.weno_pipe import weno_odep, weno_odeq

def cdm(p11, p10, p01, p00,
        q11, q10, q01, q00,
        var, lam, va, S, D, dx, dt):
    if var == 'p':
        return (p11 + p10 - p01 - p00) / (2 * dt) + va ** 2 / S * (q11 + q01 - q10 - q00) / (2 * dx)
    elif var == 'q':
        pba = (p11 + p10 + p01 + p00) / 4
        qba = (q11 + q10 + q01 + q00) / 4
        return ((q11 + q10 - q01 - q00) / (2 * dt)
                + S * (p11 + p01 - p10 - p00) / (2 * dx)
                + lam * va ** 2 * qba * Abs(qba) / (2 * D * S * pba))


def cha(p, q, p0, q0, sign, lam, va, S, D, dx):
    if sign == '+':
        return p - p0 + va / S * (q - q0) + lam * va ** 2 * dx / (4 * D * S ** 2) * (q + q0) * Abs(q + q0) / (p + p0)
    elif sign == '-':
        return p0 - p + va / S * (q - q0) + lam * va ** 2 * dx / (4 * D * S ** 2) * (q + q0) * Abs(q + q0) / (p + p0)


def cell_bd_rcst(u, Idx, sign, r, k):
    """
    u: variable
    Idx: integer part of subscript
    sign: '+' that means '+1/2' or '-' that means '-1/2'
    r: left start of stencil
    k: points in stencil
    return: reconstruction of u[Idx±1/2] based on u[i-r], ..., u[i], ..., u[i-r+k-1]
    """
    res = 0
    i = Idx
    r = Integer(r)
    i0 = Idx + Rational(1, 2) if sign == '+' else Idx - Rational(1, 2)
    for m in range(k + 1):
        for j in range(m):
            if j - r > 0:
                ubar = u[i + (j - r)]
            elif j - r < 0:
                ubar = u[i - (-j + r)]
            else:
                ubar = u[i]
            numer = Add(
                *[Mul(*[i0 - (i - r + q - Rational(1, 2)) for q in range(k + 1) if q not in [l, m]]) for l in
                  range(k + 1) if l != m])
            denom = Mul(*[(m - l) for l in range(k + 1) if l != m])
            res = res + ubar * numer / denom
    return res.expand()


def weno_rcst(u, i, order=2, pos='l'):
    """
    Reconstruction of cell boundary of u[Idx] by weno.

    The stencil reference is

    $u_{i-1/2}^{+}$------$\bar{u}_i$------$u_{i+1/2}^{-}$

    If pos=='l', return $u_{i-1/2}^{+}$. Else, return $u_{i+1/2}^{-}$.

    :param u: iVar
    :param i: idx object
    :param order: weno order
    :param pos: position of the boundary to be reconstructed
    :return:
    """
    # if not isinstance(i, idx):
    #     if not isinstance(i, (int, Integer)):
    #         raise TypeError("Subscript i must be idx object or an integer!")
    # if not isinstance(offset, Rational):
    #     raise TypeError("offset must be a rational")
    # else:
    #     if not isinstance(offset / Rational(1 / 2), Integer):
    #         raise ValueError("offset must be the integer multiplies of 1/2")
    epsilon = Rational(1, 1e6)
    one = Rational(1)

    if order == 2:

        beta0 = (u[i + one] - u[i]) ** 2
        beta1 = (u[i] - u[i - one]) ** 2

        if pos == 'r':
            gamma0 = Rational(2, 3)
            gamma1 = Rational(1, 3)
            alpha0 = gamma0 / (epsilon + beta0) ** 2
            alpha1 = gamma1 / (epsilon + beta1) ** 2
            omega0 = alpha0 / (alpha0 + alpha1)
            omega1 = alpha1 / (alpha0 + alpha1)
            u0 = cell_bd_rcst(u, i, '+', 0, 2)  # u_{i+1/2}^(0)
            u1 = cell_bd_rcst(u, i, '+', 1, 2)  # u_{i+1/2}^(1)
            umia12 = omega0 * u0 + omega1 * u1
            return umia12
        elif pos == 'l':
            tgamma0 = Rational(1, 3)
            tgamma1 = Rational(2, 3)
            talpha0 = tgamma0 / (epsilon + beta0) ** 2
            talpha1 = tgamma1 / (epsilon + beta1) ** 2
            tomega0 = talpha0 / (talpha0 + talpha1)
            tomega1 = talpha1 / (talpha0 + talpha1)
            u0 = cell_bd_rcst(u, i, '-', 0, 2)  # u_{i-1/2}^(0)
            u1 = cell_bd_rcst(u, i, '-', 1, 2)  # u_{i-1/2}^(1)
            upim12 = tomega0 * u0 + tomega1 * u1
            return upim12
    elif order == 3:

        beta0 = (Rational(13, 12) * (u[i] - 2 * u[i + one] + u[i + 2 * one]) ** 2 +
                 Rational(1, 4) * (3 * u[i] - 4 * u[i + one] + u[i + 2 * one]) ** 2)
        beta1 = (Rational(13, 12) * (u[i - one] - 2 * u[i] + u[i + one]) ** 2 +
                 Rational(1, 4) * (u[i - one] - u[i + one]) ** 2)
        beta2 = (Rational(13, 12) * (u[i - 2 * one] - 2 * u[i - one] + u[i]) ** 2 +
                 Rational(1, 4) * (u[i - one * 2] - 4 * u[i - one] + 3 * u[i]) ** 2)

        if pos == 'r':
            gamma0 = Rational(3, 10)
            gamma1 = Rational(3, 5)
            gamma2 = Rational(1, 10)
            alpha0 = gamma0 / (epsilon + beta0) ** 2
            alpha1 = gamma1 / (epsilon + beta1) ** 2
            alpha2 = gamma2 / (epsilon + beta2) ** 2
            sum_alpha = alpha0 + alpha1 + alpha2
            omega0 = alpha0 / sum_alpha
            omega1 = alpha1 / sum_alpha
            omega2 = alpha2 / sum_alpha
            u0 = cell_bd_rcst(u, i, '+', 0, 3)  # u_{i+1/2}^(0)
            u1 = cell_bd_rcst(u, i, '+', 1, 3)  # u_{i+1/2}^(1)
            u2 = cell_bd_rcst(u, i, '+', 2, 3)  # u_{i+1/2}^(2)
            umia12 = omega0 * u0 + omega1 * u1 + omega2 * u2
            return umia12
        elif pos == 'l':
            tgamma0 = Rational(1, 10)
            tgamma1 = Rational(3, 5)
            tgamma2 = Rational(3, 10)
            talpha0 = tgamma0 / (epsilon + beta0) ** 2
            talpha1 = tgamma1 / (epsilon + beta1) ** 2
            talpha2 = tgamma2 / (epsilon + beta2) ** 2
            sum_talpha = talpha0 + talpha1 + talpha2
            tomega0 = talpha0 / sum_talpha
            tomega1 = talpha1 / sum_talpha
            tomega2 = talpha2 / sum_talpha
            u0 = cell_bd_rcst(u, i, '-', 0, 3)  # u_{i-1/2}^(0)
            u1 = cell_bd_rcst(u, i, '-', 1, 3)  # u_{i-1/2}^(1)
            u2 = cell_bd_rcst(u, i, '-', 2, 3)  # u_{i-1/2}^(2)
            upim12 = tomega0 * u0 + tomega1 * u1 + tomega2 * u2
            return upim12


def mol_weno_q_eqn_rhs(p, q, S, va, lam, D, dx, l_idx, r_idx):
    l = l_idx
    r = r_idx
    i = idx('i')
    hpr = 1 / 2 * (S * weno_rcst(p, i, order=2, pos='r') + va * weno_rcst(q, i, order=2, pos='r'))
    hpl = 1 / 2 * (S * weno_rcst(p, i, order=2, pos='l') - va * weno_rcst(q, i, order=2, pos='l'))
    # f_{i+1/2}^-
    h1 = hpr.subs({p[i]: p[l:r],
                   p[i + 1]: p[l + 1:r + 1],
                   p[i - 1]: p[l - 1:r - 1],
                   q[i]: q[l:r],
                   q[i + 1]: q[l + 1:r + 1],
                   q[i - 1]: q[l - 1:r - 1]})
    # f_{i-1/2}^+
    h2 = hpl.subs({p[i]: p[l:r],
                   p[i + 1]: p[l + 1:r + 1],
                   p[i - 1]: p[l - 1:r - 1],
                   q[i]: q[l:r],
                   q[i + 1]: q[l + 1:r + 1],
                   q[i - 1]: q[l - 1:r - 1]})
    # f_{i-1/2}^-
    h3 = hpr.subs({p[i]: p[l - 1:r - 1],
                   p[i + 1]: p[l:r],
                   p[i - 1]: p[l - 2:r - 2],
                   q[i]: q[l - 1:r - 1],
                   q[i + 1]: q[l:r],
                   q[i - 1]: q[l - 2:r - 2]})
    # f_{i+1/2}^+
    h4 = hpl.subs({p[i]: p[l + 1:r + 1],
                   p[i + 1]: p[l + 2:r + 2],
                   p[i - 1]: p[l:r],
                   q[i]: q[l + 1:r + 1],
                   q[i + 1]: q[l + 2:r + 2],
                   q[i - 1]: q[l:r]})
    return (-(h1 + h4 - h2 - h3) / dx - lam * va ** 2 * q[l:r] * Abs(q[l:r]) / (
            2 * D * S * p[l:r]))


def mol_weno_q_eqn_rhs1(p_list, q_list, S, va, lam, D, dx):
    p_list = [arg.symbol if isinstance(arg, Var) else arg for arg in p_list]
    q_list = [arg.symbol if isinstance(arg, Var) else arg for arg in q_list]
    pm2, pm1, p0, pp1, pp2 = p_list
    qm2, qm1, q0, qp1, qp2 = q_list
    i = idx('i')
    p = iVar('p')
    q = iVar('q')
    hpr = 1 / 2 * (S * weno_rcst(p, i, order=2, pos='r') + va * weno_rcst(q, i, order=2, pos='r'))
    hpl = 1 / 2 * (S * weno_rcst(p, i, order=2, pos='l') - va * weno_rcst(q, i, order=2, pos='l'))
    # f_{i+1/2}^-
    h1 = hpr.subs({p[i]: p0,
                   p[i + 1]: pp1,
                   p[i - 1]: pm1,
                   q[i]: q0,
                   q[i + 1]: qp1,
                   q[i - 1]: qm1})
    # f_{i-1/2}^+
    h2 = hpl.subs({p[i]: p0,
                   p[i + 1]: pp1,
                   p[i - 1]: pm1,
                   q[i]: q0,
                   q[i + 1]: qp1,
                   q[i - 1]: qm1})
    # f_{i-1/2}^-
    h3 = hpr.subs({p[i]: pm1,
                   p[i + 1]: p0,
                   p[i - 1]: pm2,
                   q[i]: qm1,
                   q[i + 1]: q0,
                   q[i - 1]: qm2})
    # f_{i+1/2}^+
    h4 = hpl.subs({p[i]: pp1,
                   p[i + 1]: pp2,
                   p[i - 1]: p0,
                   q[i]: qp1,
                   q[i + 1]: qp2,
                   q[i - 1]: q0})
    return (-(h1 + h4 - h2 - h3) / dx - lam * va ** 2 * q0 * Abs(q0) / (
            2 * D * S * p0))


def mol_eno_q_eqn(p, q, S, va, lam, D, dx, M, i):
    """
        i == 4 the first one

        i == 3 the second one

        i == 1 the last but one

        i == 0 the last one

    """
    match i:
        case 1:
            # f_{M-1/2}^-
            p2 = cell_bd_rcst(p, M - 1, '+', 3, 5)  # p_{M-1/2}^-
            q2 = cell_bd_rcst(q, M - 1, '+', 3, 5)  # q_{M-1/2}^-
            h1 = 1 / 2 * (S * p2 + va * q2)
            # f_{M-3/2}^+
            p1 = cell_bd_rcst(p, M - 1, '-', 4, 5)  # p_{M-3/2}^+
            q1 = cell_bd_rcst(q, M - 1, '-', 4, 5)  # q_{M-3/2}^+
            h2 = 1 / 2 * (S * p1 - va * q1)
            # f_{M-3/2}^-
            p2 = cell_bd_rcst(p, M - 2, '+', 3, 5)  # p_{M-3/2}^-
            q2 = cell_bd_rcst(q, M - 2, '+', 3, 5)  # q_{M-3/2}^-
            h3 = 1 / 2 * (S * p2 + va * q2)
            # f_{M-1/2}^+
            p1 = cell_bd_rcst(p, M, '-', 4, 5)  # p_{M-3/2}^+
            q1 = cell_bd_rcst(q, M, '-', 4, 5)  # q_{M-3/2}^+
            h4 = 1 / 2 * (S * p1 - va * q1)
            Source = lam * va ** 2 * q[M - 1] * Abs(q[M - 1]) / (2 * D * S * p[M - 1])
        case 0:
            # f_{M+1/2}^-
            p2 = cell_bd_rcst(p, M, '+', 4, 5)  # p_{M+1/2}^-
            q2 = cell_bd_rcst(q, M, '+', 4, 5)  # q_{M+1/2}^-
            h1 = 1 / 2 * (S * p2 + va * q2)
            # f_{M-1/2}^+
            p1 = cell_bd_rcst(p, M, '-', 4, 5)  # p_{M-1/2}^+
            q1 = cell_bd_rcst(q, M, '-', 4, 5)  # q_{M-1/2}^+
            h2 = 1 / 2 * (S * p1 - va * q1)
            # f_{M-1/2}^-
            h3 = h2
            # f_{M+1/2}^+
            h4 = h1
            Source = lam * va ** 2 * q[M] * Abs(q[M]) / (2 * D * S * p[M])
        case 3:
            # f_{3/2}^-
            p2 = cell_bd_rcst(p, 1, '+', 0, 5)  # p_{3/2}^-
            q2 = cell_bd_rcst(q, 1, '+', 0, 5)  # q_{3/2}^-
            h1 = 1 / 2 * (S * p2 + va * q2)
            # f_{1/2}^+
            p1 = cell_bd_rcst(p, 1, '-', 1, 5)  # p_{1/2}^+
            q1 = cell_bd_rcst(q, 1, '-', 1, 5)  # q_{1/2}^+
            h2 = 1 / 2 * (S * p1 - va * q1)
            # f_{1/2}^-
            p2 = cell_bd_rcst(p, 0, '+', 0, 5)  # p_{1/2}^-
            q2 = cell_bd_rcst(q, 0, '+', 0, 5)  # q_{1/2}^-
            h3 = 1 / 2 * (S * p2 + va * q2)
            # f_{3/2}^+
            p1 = cell_bd_rcst(p, 2, '-', 1, 5)  # p_{3/2}^+
            q1 = cell_bd_rcst(q, 2, '-', 1, 5)  # q_{3/2}^+
            h4 = 1 / 2 * (S * p1 - va * q1)
            Source = lam * va ** 2 * q[1] * Abs(q[1]) / (2 * D * S * p[1])
        case 4:
            # f_{1/2}^-
            p2 = cell_bd_rcst(p, 0, '+', 0, 5)  # p_{1/2}^-
            q2 = cell_bd_rcst(q, 0, '+', 0, 5)  # q_{1/2}^-
            h1 = 1 / 2 * (S * p2 + va * q2)
            # f_{-1/2}^+
            p1 = cell_bd_rcst(p, 0, '-', 0, 5)  # p_{1/2}^+
            q1 = cell_bd_rcst(q, 0, '-', 0, 5)  # q_{1/2}^+
            h2 = 1 / 2 * (S * p1 - va * q1)
            # f_{-1/2}^-
            h3 = h2
            # f_{1/2}^+
            h4 = h1
            Source = lam * va ** 2 * q[0] * Abs(q[0]) / (2 * D * S * p[0])
        case _:
            raise NotImplementedError(f"Case {i} not implemented!")
    return -(h1 + h4 - h2 - h3) / dx - Source


def mol_eno_p_eqn(p, q, S, va, dx, M, i):
    """
        i == 4 the first one

        i == 3 the second one

        i == 1 the last but one

        i == 0 the last one

    """
    match i:
        case 1:
            # f_{M-1/2}^-
            p2 = cell_bd_rcst(p, M - 1, '+', 3, 5)  # p_{M-1/2}^-
            q2 = cell_bd_rcst(q, M - 1, '+', 3, 5)  # q_{M-1/2}^-
            h1 = 1 / 2 * (va ** 2 / S * q2 + va * p2)
            # f_{M-3/2}^+
            p1 = cell_bd_rcst(p, M - 1, '-', 4, 5)  # p_{M-3/2}^+
            q1 = cell_bd_rcst(q, M - 1, '-', 4, 5)  # q_{M-3/2}^+
            h2 = 1 / 2 * (va ** 2 / S * q1 - va * p1)
            # f_{M-3/2}^-
            p2 = cell_bd_rcst(p, M - 2, '+', 3, 5)  # p_{M-3/2}^-
            q2 = cell_bd_rcst(q, M - 2, '+', 3, 5)  # q_{M-3/2}^-
            h3 = 1 / 2 * (va ** 2 / S * q2 + va * p2)
            # f_{M-1/2}^+
            p1 = cell_bd_rcst(p, M, '-', 4, 5)  # p_{M-3/2}^+
            q1 = cell_bd_rcst(q, M, '-', 4, 5)  # q_{M-3/2}^+
            h4 = 1 / 2 * (va ** 2 / S * q1 - va * p1)
        case 0:
            # f_{M+1/2}^-
            p2 = cell_bd_rcst(p, M, '+', 4, 5)  # p_{M+1/2}^-
            q2 = cell_bd_rcst(q, M, '+', 4, 5)  # q_{M+1/2}^-
            h1 = 1 / 2 * (va ** 2 / S * q2 + va * p2)
            # f_{M-1/2}^+
            p1 = cell_bd_rcst(p, M, '-', 5, 5)  # p_{M-1/2}^+
            q1 = cell_bd_rcst(q, M, '-', 5, 5)  # q_{M-1/2}^+
            h2 = 1 / 2 * (va ** 2 / S * q1 - va * p1)
            # f_{M-1/2}^-
            h3 = h2
            # f_{M+1/2}^+
            h4 = h1
        case 3:
            # f_{3/2}^-
            p2 = cell_bd_rcst(p, 1, '+', 0, 5)  # p_{3/2}^-
            q2 = cell_bd_rcst(q, 1, '+', 0, 5)  # q_{3/2}^-
            h1 = 1 / 2 * (va ** 2 / S * q2 + va * p2)
            # f_{1/2}^+
            p1 = cell_bd_rcst(p, 1, '-', 1, 5)  # p_{1/2}^+
            q1 = cell_bd_rcst(q, 1, '-', 1, 5)  # q_{1/2}^+
            h2 = 1 / 2 * (va ** 2 / S * q1 - va * p1)
            # f_{1/2}^-
            p2 = cell_bd_rcst(p, 0, '+', 0, 5)  # p_{1/2}^-
            q2 = cell_bd_rcst(q, 0, '+', 0, 5)  # q_{1/2}^-
            h3 = 1 / 2 * (va ** 2 / S * q2 + va * p2)
            # f_{3/2}^+
            p1 = cell_bd_rcst(p, 2, '-', 1, 5)  # p_{3/2}^+
            q1 = cell_bd_rcst(q, 2, '-', 1, 5)  # q_{3/2}^+
            h4 = 1 / 2 * (va ** 2 / S * q1 - va * p1)
        case 4:
            # f_{1/2}^-
            p2 = cell_bd_rcst(p, 0, '+', -1, 5)  # p_{1/2}^-
            q2 = cell_bd_rcst(q, 0, '+', -1, 5)  # q_{1/2}^-
            h1 = 1 / 2 * (va ** 2 / S * q2 + va * p2)
            # f_{-1/2}^+
            p1 = cell_bd_rcst(p, 0, '-', 0, 5)  # p_{1/2}^+
            q1 = cell_bd_rcst(q, 0, '-', 0, 5)  # q_{1/2}^+
            h2 = 1 / 2 * (va ** 2 / S * q1 - va * p1)
            # f_{-1/2}^-
            h3 = h2
            # f_{1/2}^+
            h4 = h1
        case _:
            raise NotImplementedError(f"Case {i} not implemented!")
    return -(h1 + h4 - h2 - h3) / dx


def mol_weno_p_eqn_rhs(p, q, S, va, dx, l_idx, r_idx):
    l = l_idx
    r = r_idx
    i = idx('i')
    hql = 1 / 2 * (va ** 2 / S * weno_rcst(q, i, order=2, pos='l') - va * weno_rcst(p, i, order=2, pos='l'))
    hqr = 1 / 2 * (va ** 2 / S * weno_rcst(q, i, order=2, pos='r') + va * weno_rcst(p, i, order=2, pos='r'))
    # f_{i+1/2}^-
    h1 = hqr.subs({p[i]: p[l:r],
                   p[i + 1]: p[l + 1:r + 1],
                   p[i - 1]: p[l - 1:r - 1],
                   q[i]: q[l:r],
                   q[i + 1]: q[l + 1:r + 1],
                   q[i - 1]: q[l - 1:r - 1]})
    # f_{i-1/2}^+
    h2 = hql.subs({p[i]: p[l:r],
                   p[i + 1]: p[l + 1:r + 1],
                   p[i - 1]: p[l - 1:r - 1],
                   q[i]: q[l:r],
                   q[i + 1]: q[l + 1:r + 1],
                   q[i - 1]: q[l - 1:r - 1]})
    # f_{i-1/2}^-
    h3 = hqr.subs({p[i]: p[l - 1:r - 1],
                   p[i + 1]: p[l:r],
                   p[i - 1]: p[l - 2:r - 2],
                   q[i]: q[l - 1:r - 1],
                   q[i + 1]: q[l:r],
                   q[i - 1]: q[l - 2:r - 2]})
    # f_{i+1/2}^+
    h4 = hql.subs({p[i]: p[l + 1:r + 1],
                   p[i + 1]: p[l + 2:r + 2],
                   p[i - 1]: p[l:r],
                   q[i]: q[l + 1:r + 1],
                   q[i + 1]: q[l + 2:r + 2],
                   q[i - 1]: q[l:r]})
    return -(h1 + h4 - h2 - h3) / dx


def mol_weno_p_eqn_rhs1(p_list, q_list, S, va, dx):
    p_list = [arg.symbol if isinstance(arg, Var) else arg for arg in p_list]
    q_list = [arg.symbol if isinstance(arg, Var) else arg for arg in q_list]
    pm2, pm1, p0, pp1, pp2 = p_list
    qm2, qm1, q0, qp1, qp2 = q_list
    i = idx('i')
    p = iVar('p')
    q = iVar('q')
    hql = 1 / 2 * (va ** 2 / S * weno_rcst(q, i, order=2, pos='l') - va * weno_rcst(p, i, order=2, pos='l'))
    hqr = 1 / 2 * (va ** 2 / S * weno_rcst(q, i, order=2, pos='r') + va * weno_rcst(p, i, order=2, pos='r'))
    # f_{i+1/2}^-
    h1 = hqr.subs({p[i]: p0,
                   p[i + 1]: pp1,
                   p[i - 1]: pm1,
                   q[i]: q0,
                   q[i + 1]: qp1,
                   q[i - 1]: qm1})
    # f_{i-1/2}^+
    h2 = hql.subs({p[i]: p0,
                   p[i + 1]: pp1,
                   p[i - 1]: pm1,
                   q[i]: q0,
                   q[i + 1]: qp1,
                   q[i - 1]: qm1})
    # f_{i-1/2}^-
    h3 = hqr.subs({p[i]: pm1,
                   p[i + 1]: p0,
                   p[i - 1]: pm2,
                   q[i]: qm1,
                   q[i + 1]: q0,
                   q[i - 1]: qm2})
    # f_{i+1/2}^+
    h4 = hql.subs({p[i]: pp1,
                   p[i + 1]: pp2,
                   p[i - 1]: p0,
                   q[i]: qp1,
                   q[i + 1]: qp2,
                   q[i - 1]: q0})
    return -(h1 + h4 - h2 - h3) / dx


def mol_eno_p_eqn_rhs1(p_list, q_list, S, va, dx):
    p_list = [arg.symbol if isinstance(arg, Var) else arg for arg in p_list]
    q_list = [arg.symbol if isinstance(arg, Var) else arg for arg in q_list]
    pm2, pm1, p0, pp1, pp2 = p_list
    qm2, qm1, q0, qp1, qp2 = q_list
    i = idx('i')
    p = iVar('p')
    q = iVar('q')
    hql = 1 / 2 * (va ** 2 / S * weno_rcst(q, i, order=2, pos='l') - va * weno_rcst(p, i, order=2, pos='l'))
    hqr = 1 / 2 * (va ** 2 / S * weno_rcst(q, i, order=2, pos='r') + va * weno_rcst(p, i, order=2, pos='r'))
    # f_{i+1/2}^-
    h1 = hqr.subs({p[i]: p0,
                   p[i + 1]: pp1,
                   p[i - 1]: pm1,
                   q[i]: q0,
                   q[i + 1]: qp1,
                   q[i - 1]: qm1})
    # f_{i-1/2}^+
    h2 = hql.subs({p[i]: p0,
                   p[i + 1]: pp1,
                   p[i - 1]: pm1,
                   q[i]: q0,
                   q[i + 1]: qp1,
                   q[i - 1]: qm1})
    # f_{i-1/2}^-
    h3 = hqr.subs({p[i]: pm1,
                   p[i + 1]: p0,
                   p[i - 1]: pm2,
                   q[i]: qm1,
                   q[i + 1]: q0,
                   q[i - 1]: qm2})
    # f_{i+1/2}^+
    h4 = hql.subs({p[i]: pp1,
                   p[i + 1]: pp2,
                   p[i - 1]: p0,
                   q[i]: qp1,
                   q[i + 1]: qp2,
                   q[i - 1]: q0})
    return -(h1 + h4 - h2 - h3) / dx


def mol_tvd1_p_eqn_rhs1(p_list, q_list, S, va, dx):
    p_list = [arg.symbol if isinstance(arg, Var) else arg for arg in p_list]
    q_list = [arg.symbol if isinstance(arg, Var) else arg for arg in q_list]
    pm1, p0, pp1 = p_list
    qm1, q0, qp1 = q_list
    return -va ** 2 / S * (qp1 - qm1) / (2 * dx)


def mol_tvd1_q_eqn_rhs1(p_list, q_list, S, va, lam, D, dx):
    p_list = [arg.symbol if isinstance(arg, Var) else arg for arg in p_list]
    q_list = [arg.symbol if isinstance(arg, Var) else arg for arg in q_list]
    pm1, p0, pp1 = p_list
    qm1, q0, qp1 = q_list
    return -S * (pp1 - pm1) / (2 * dx) - lam * va ** 2 * q0 * Abs(q0) / (
            2 * D * S * p0)


def mol_tvd1_p_eqn_rhs0(p_list, q_list, S, va, dx):
    p_list = [arg.symbol if isinstance(arg, Var) else arg for arg in p_list]
    q_list = [arg.symbol if isinstance(arg, Var) else arg for arg in q_list]
    pm1, p0, pp1 = p_list
    qm1, q0, qp1 = q_list
    return -va ** 2 / S * (qp1 - qm1) / (2 * dx) + va * (pp1 - 2 * p0 + pm1) / (2 * dx)


def mol_tvd1_q_eqn_rhs0(p_list, q_list, S, va, lam, D, dx):
    p_list = [arg.symbol if isinstance(arg, Var) else arg for arg in p_list]
    q_list = [arg.symbol if isinstance(arg, Var) else arg for arg in q_list]
    pm1, p0, pp1 = p_list
    qm1, q0, qp1 = q_list
    return -S * (pp1 - pm1) / (2 * dx) + va * (qp1 - 2 * q0 + qm1) / (2 * dx) - lam * va ** 2 * q0 * Abs(q0) / (
            2 * D * S * p0)


from Solverz import minmod


def ux(theta, um1, u, up1, dx):
    return minmod(theta * (u - um1) / dx, (up1 - um1) / (2 * dx), theta * (up1 - u) / dx)


def mol_tvd2_p_eqn_rhs(p_list, q_list, S, va, dx):

    pm2, pm1, p0, pp1, pp2 = p_list
    qm2, qm1, q0, qp1, qp2 = q_list

    theta = Param('theta', 1)

    def f(p, q):
        return va ** 2 / S * q

    def Source(p, q):
        return Integer(0)

    # u_{j+1/2}^+
    p1 = pp1 - dx / 2 * ux(theta, p0, pp1, pp2, dx)
    q1 = qp1 - dx / 2 * ux(theta, q0, qp1, qp2, dx)

    # u_{j+1/2}^-
    p2 = p0 + dx / 2 * ux(theta, pm1, p0, pp1, dx)
    q2 = q0 + dx / 2 * ux(theta, qm1, q0, qp1, dx)

    # u_{j-1/2}^+
    p3 = p0 - dx / 2 * ux(theta, pm1, p0, pp1, dx)
    q3 = q0 - dx / 2 * ux(theta, qm1, q0, qp1, dx)

    # u_{j-1/2}^-
    p4 = pm1 + dx / 2 * ux(theta, pm2, pm1, p0, dx)
    q4 = qm1 + dx / 2 * ux(theta, qm2, qm1, q0, dx)

    Hp = (f(p1, q1) + f(p2, q2)) / 2 - va / 2 * (p1 - p2)
    Hm = (f(p3, q3) + f(p4, q4)) / 2 - va / 2 * (p3 - p4)

    return -(Hp - Hm) / dx + Source(p0, q0)


def mol_tvd2_q_eqn_rhs(p_list, q_list, S, va, lam, D, dx):

    pm2, pm1, p0, pp1, pp2 = p_list
    qm2, qm1, q0, qp1, qp2 = q_list

    theta = Param('theta', 1)

    def f(p, q):
        return S * p

    def Source(p, q):
        return -lam * va ** 2 * q * Abs(q) / (2 * D * S * p)

    # u_{j+1/2}^+
    p1 = pp1 - dx / 2 * ux(theta, p0, pp1, pp2, dx)
    q1 = qp1 - dx / 2 * ux(theta, q0, qp1, qp2, dx)

    # u_{j+1/2}^-
    p2 = p0 + dx / 2 * ux(theta, pm1, p0, pp1, dx)
    q2 = q0 + dx / 2 * ux(theta, qm1, q0, qp1, dx)

    # u_{j-1/2}^+
    p3 = p0 - dx / 2 * ux(theta, pm1, p0, pp1, dx)
    q3 = q0 - dx / 2 * ux(theta, qm1, q0, qp1, dx)

    # u_{j-1/2}^-
    p4 = pm1 + dx / 2 * ux(theta, pm2, pm1, p0, dx)
    q4 = qm1 + dx / 2 * ux(theta, qm2, qm1, q0, dx)

    Hp = (f(p1, q1) + f(p2, q2)) / 2 - va / 2 * (q1 - q2)
    Hm = (f(p3, q3) + f(p4, q4)) / 2 - va / 2 * (q3 - q4)

    return -(Hp - Hm) / dx + Source(p0, q0)
