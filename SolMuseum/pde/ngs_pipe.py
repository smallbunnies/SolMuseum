from .util import *


def ngs_pipe(p: Var,
             q: Var,
             lam,
             va,
             D,
             S,
             dx,
             dt,
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
                p_i^{j+1}-p_{i-1}^j+\frac{c}{S}\left(q_i^{j+1}-q_{i-1}^j\right)+\frac{\lambda c^2 \Delta x}{4 D S^2} \frac{\left(q_i^{j+1}+q_{i-1}^j\right)\left|q_i^{j+1}+q_{i-1}^j\right|}{p_i^{j+1}+p_{i-1}^j}&=0,\quad 1\leq i\leq M,\\
                p_{i+1}^j-p_i^{j+1}+\frac{c}{S}\left(q_i^{j+1}-q_{i+1}^j\right)+\frac{\lambda c^2 \Delta x}{4 D S^2} \frac{\left(q_i^{j+1}+q_{i+1}^j\right)\left|q_i^{j+1}+q_{i+1}^j\right|}{p_i^{j+1}+p_{i+1}^j}&=0,\quad 0\leq i\leq M-1.

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
            p0 = AliasVar(p.name, init=p)
            q0 = AliasVar(q.name, init=q)
            artifact[p0.name] = p0
            artifact[q0.name] = q0
            rhs = (p[1:M + 1] - p0[1:M + 1]) / dt + va ** 2 / S * (q[1:M + 1] - q[0:M]) / dx
            artifact[f'euler{pipe_name}_p'] = Eqn(f'euler{pipe_name}_p', rhs)
            rhs = (q[1:M + 1] - q0[1:M + 1]) / dt + S * (p[1:M + 1] - p[0:M]) / dx + lam * va ** 2 * q[1:M + 1] * Abs(
                q[1:M + 1]) / (2 * D * S * p[1:M + 1])
            artifact[f'euler{pipe_name}_q'] = Eqn(f'euler{pipe_name}_q', rhs)

        case 'cdm':
            p0 = AliasVar(p.name, init=p)
            q0 = AliasVar(q.name, init=q)
            artifact[p0.name] = p0
            artifact[q0.name] = q0
            rhs = ((p[1:M + 1] + p[0:M] - p0[1:M + 1] - p0[0:M]) / (2 * dt)
                   + va ** 2 / S * (q[1:M + 1] + q0[1:M + 1] - q[0:M] - q0[0:M]) / (2 * dx))
            artifact[f'cdm{pipe_name}_p'] = Eqn(f'cdm{pipe_name}_p', rhs)
            pba = (p[1:M + 1] + p[0:M] + p0[1:M + 1] + p0[0:M]) / 4
            qba = (q[1:M + 1] + q[0:M] + q0[1:M + 1] + q0[0:M]) / 4
            rhs = ((q[1:M + 1] + q[0:M] - q0[1:M + 1] - q0[0:M]) / (2 * dt)
                   + S * (p[1:M + 1] + p0[1:M + 1] - p[0:M] - p0[0:M]) / (2 * dx)
                   + lam * va ** 2 * qba * Abs(qba) / (2 * D * S * pba))
            artifact[f'cdm{pipe_name}_q'] = Eqn(f'cdm{pipe_name}_q', rhs)

        case 'cha':
            p0 = AliasVar(p.name, init=p)
            q0 = AliasVar(q.name, init=q)
            artifact[p0.name] = p0
            artifact[q0.name] = q0
            p_ = p[1:M + 1]
            q_ = q[1:M + 1]
            p_0 = p0[0:M]
            q_0 = q0[0:M]
            rhs = cha(p_, q_, p_0, q_0, '+', lam, va, S, D, dx)
            artifact[f'cha_{pipe_name}_pos'] = Eqn(f'cha_{pipe_name}_pos', rhs)

            p_ = p[0:M]
            q_ = q[0:M]
            p_0 = p0[1:M + 1]
            q_0 = q0[1:M + 1]
            rhs = cha(p_, q_, p_0, q_0, '-', lam, va, S, D, dx)
            artifact[f'cha_{pipe_name}_neg'] = Eqn(f'cha_{pipe_name}_neg', rhs)
        case 'kt1':
            rhs = mol_tvd1_q_eqn_rhs0([p[0:M - 1], p[1:M], p[2:M + 1]],
                                      [q[0:M - 1], q[1:M], q[2:M + 1]],
                                      S,
                                      va,
                                      lam,
                                      D,
                                      dx)
            artifact['q_' + pipe_name + '_eqn'] = Ode(f'kt1-q{pipe_name}',
                                                      rhs,
                                                      q[1:M])

            rhs = mol_tvd1_p_eqn_rhs0([p[0:M - 1], p[1:M], p[2:M + 1]],
                                      [q[0:M - 1], q[1:M], q[2:M + 1]],
                                      S,
                                      va,
                                      dx)
            artifact['p_' + pipe_name + '_eqn'] = Ode(f'kt1-p{pipe_name}',
                                                      rhs,
                                                      p[1:M])

            artifact['p_' + pipe_name + 'bd1'] = Eqn(p.name + 'bd1',
                                                     S * p[M] + va * q[M] + S * p[M - 2] + va * q[
                                                         M - 2] - 2 * (
                                                             S * p[M - 1] + va * q[M - 1]))
            artifact['q_' + pipe_name + 'bd2'] = Eqn(q.name + 'bd2',
                                                     S * p[2] - va * q[2] + S * p[0] - va * q[0] - 2 * (
                                                             S * p[1] - va * q[1]))

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
