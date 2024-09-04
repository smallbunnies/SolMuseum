from Solverz import Eqn, Ode, AliasVar
from Solverz import iVar, idx, Var, Abs
from Solverz.utilities.type_checker import is_integer, is_number
from sympy import Add, Mul, Rational, Integer


def ngs_pipe(p: Var,
             q: Var,
             lam,
             va,
             D,
             S,
             dx,
             M,
             pipe_name: str,
             method='euler'):
    r"""
    This function model and discretize natural gas equations

        .. math::
            \pdv{p}{t}+\frac{c^2}{S}\pdv{q}{x}=&0,\\
            \pdv{q}{t}+S\pdv{p}{x}+\frac{\lambda c^2q|q|}{2DSp}=&0

    Parameters
    ==========

    p : Var
        The pressure w.r.t. x and t

    q : Var
        The mass flow w.r.t. x and t

    lam : Param or Number
        The friction $\lambda$

    va : Param or Number
        The speed of sound

    D : Param or Number
        The pipe diameter

    S : Param or Number
        The cross-section area

    dx : Param or Number
        The spatial difference step size

    M : Param or Number
        The friction $\lambda$

    pipe_name : str
        The name of the pipe

    method : str

        'cha' - The method of characteristics

            .. math ::
                p_i^{j+1}-p_{i-1}^j+\frac{c}{S}\left(q_i^{j+1}-q_{i-1}^j\right)+\frac{\lambda c^2 \Delta x}{4 D S^2} \frac{\left(q_i^{j+1}+q_{i-1}^j\right)^2}{p_i^{j+1}+p_{i-1}^j}&=0,\quad 1\leq i\leq M,\\
                p_{i+1}^j-p_i^{j+1}+\frac{c}{S}\left(q_i^{j+1}-q_{i+1}^j\right)+\frac{\lambda c^2 \Delta x}{4 D S^2} \frac{\left(q_i^{j+1}+q_{i+1}^j\right)^2}{p_i^{j+1}+p_{i+1}^j}&=0,\quad 0\leq i\leq M-1.

        'weno' - The WENO semi-discretization

            .. math ::
                \pdv{u_j}{t}=-\frac{1}{\Delta x}\qty(\hat{f}_{j+1/2}-\hat{f}_{j-1/2})+S(u_j)

            where $\hat{f}_{j+1/2}$ and $\hat{f}_{j-1/2}$ are reconstructed by the weighted essentially non-oscillatory
            scheme [1]_.

    Returns
    =======

    artifact : dict()
        The dict of derived equations and variables.

    References
    ==========

    .. [1] C.-W. Shu, “Essentially non-oscillatory and weighted essentially non- oscillatory schemes for hyperbolic conservation laws,” in Advanced Numerical Approximation of Nonlinear Hyperbolic Equations: Lectures given at the 2nd Session of the Centro Internazionale Matematico Estivo (C.I.M.E.) held in Cetraro, Italy, June 23–28, 1997, A. Quarteroni, Ed. Berlin, Heidelberg: Springer Berlin Heidelberg, 1998, pp. 325–432.

    """

    if not is_integer(M):
        raise TypeError(f'M is {type(M)} instead of integer')

    if not is_number(dx):
        raise TypeError(f'dx is {type(dx)} instead of number')

    artifact = dict()

    match method:
        case 'euler':
            pass
        case 'cdm':
            pass
        case 'cha':
            p0 = AliasVar(p.name, init=p)
            q0 = AliasVar(q.name, init=q)
            artifact[p0.name] = p0
            artifact[q0.name] = q0
            p_ = p[1:M + 1]
            q_ = q[1:M + 1]
            p_0 = p0[0:M]
            q_0 = q0[0:M]
            rhs = p_ - p_0 + va / S * (q_ - q_0) + lam * va ** 2 * dx / (4 * D * S ** 2) * (q_ + q_0) * Abs(
                q_ + q_0) / (p_ + p_0)
            artifact[f'cha_{pipe_name}_pos'] = Eqn(f'cha_{pipe_name}_pos', rhs)

            p_ = p[0:M]
            q_ = q[0:M]
            p_0 = p0[1:M + 1]
            q_0 = q0[1:M + 1]
            rhs = p_0 - p_ + va / S * (q_ - q_0) + lam * va ** 2 * dx / (4 * D * S ** 2) * (q_ + q_0) * Abs(
                q_ + q_0) / (p_ + p_0)
            artifact[f'cha_{pipe_name}_neg'] = Eqn(f'cha_{pipe_name}_neg', rhs)

            return artifact
        case 'weno':
            rhs = mol_weno_q_eqn_rhs(p, q, S, va, lam, D, dx, 2, M - 1)
            artifact['q_' + pipe_name + '_eqn1'] = Ode(f'weno3-q{pipe_name}_1',
                                                   rhs,
                                                   q[2:M - 1])
            rhs = mol_tvd1_q_eqn_rhs1([p[0], p[1], p[2]],
                                      [q[0], q[1], q[2]],
                                      S,
                                      va,
                                      lam,
                                      D,
                                      dx)
            artifact['q_' + pipe_name + '_eqn2'] = Ode(f'weno3-q{pipe_name}_2',
                                                   rhs,
                                                   q[1])
            rhs = mol_tvd1_q_eqn_rhs1([p[M - 2], p[M - 1], p[M]],
                                      [q[M - 2], q[M - 1], q[M]],
                                      S,
                                      va,
                                      lam,
                                      D,
                                      dx)
            artifact['q_' + pipe_name + '_eqn3'] = Ode(f'weno3-q{pipe_name}_3',
                                                   rhs,
                                                   q[M - 1])
            rhs = mol_weno_p_eqn_rhs(p, q, S, va, dx, 2, M - 1)
            artifact['p_' + pipe_name + '_eqn1'] = Ode(f'weno3-p{pipe_name}_1',
                                                   rhs,
                                                   p[2:M - 1])
            rhs = mol_tvd1_p_eqn_rhs1([p[0], p[1], p[2]],
                                      [q[0], q[1], q[2]],
                                      S,
                                      va,
                                      dx)
            artifact['p_' + pipe_name + '_eqn2'] = Ode(f'weno3-p{pipe_name}_2',
                                                   rhs,
                                                   p[1])
            rhs = mol_tvd1_p_eqn_rhs1([p[M - 2], p[M - 1], p[M]],
                                      [q[M - 2], q[M - 1], q[M]],
                                      S,
                                      va,
                                      dx)
            artifact['p_' + pipe_name + '_eqn3'] = Ode(f'weno3-p{pipe_name}_3',
                                                   rhs,
                                                   p[M - 1])
            artifact['p_' + pipe_name + 'bd1'] = Eqn(p.name + 'bd1',
                                                 S * p[M] + va * q[M] + S * p[M - 2] + va * q[
                                                     M - 2] - 2 * (
                                                         S * p[M - 1] + va * q[M - 1]))
            artifact['q_' + pipe_name + 'bd2'] = Eqn(q.name + 'bd2',
                                                 S * p[2] - va * q[2] + S * p[0] - va * q[0] - 2 * (
                                                         S * p[1] - va * q[1]))
            return artifact


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
