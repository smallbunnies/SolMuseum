from .util import *

__all__ = ['rupture_ngs_pipe_cdm', 'leakage_ngs_pipe_cdm']


def rupture_ngs_pipe_cdm(p: Var,
                         q: Var,
                         lam,
                         va,
                         D,
                         S,
                         dx,
                         dt,
                         M,
                         pipe_name: str,
                         idx_leak):
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

    qleak1 = Var(f'q_{pipe_name}_leak1', init=q[idx_leak - 1])
    qleak2 = Var(f'q_{pipe_name}_leak2', init=q[idx_leak + 1])
    qleak1_0 = AliasVar(qleak1.name, init=qleak1)
    qleak2_0 = AliasVar(qleak2.name, init=qleak2)

    artifact[qleak1.name] = qleak1
    artifact[qleak2.name] = qleak2
    artifact[qleak1_0.name] = qleak1_0
    artifact[qleak2_0.name] = qleak2_0

    p0 = AliasVar(p.name, init=p)
    q0 = AliasVar(q.name, init=q)
    artifact[p0.name] = p0
    artifact[q0.name] = q0

    # 1<=i<=idx_leak-1
    rhs = cdm(p[1:idx_leak], p[0:idx_leak - 1], p0[1:idx_leak], p0[0:idx_leak - 1],
              q[1:idx_leak], q[0:idx_leak - 1], q0[1:idx_leak], q0[0:idx_leak - 1],
              'p', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_p_0'] = Eqn(f'cdm_{pipe_name}_p_0',
                                           rhs)

    rhs = cdm(p[1:idx_leak], p[0:idx_leak - 1], p0[1:idx_leak], p0[0:idx_leak - 1],
              q[1:idx_leak], q[0:idx_leak - 1], q0[1:idx_leak], q0[0:idx_leak - 1],
              'q', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_q_0'] = Eqn(f'cdm_{pipe_name}_q_0',
                                           rhs)

    # i=idx_leak
    rhs = cdm(p[idx_leak], p[idx_leak - 1], p0[idx_leak], p0[idx_leak - 1],
              qleak1, q[idx_leak - 1], qleak1_0, q0[idx_leak - 1],
              'p', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_p_1'] = Eqn(f'cdm_{pipe_name}_p_1', rhs)

    rhs = cdm(p[idx_leak], p[idx_leak - 1], p0[idx_leak], p0[idx_leak - 1],
              qleak1, q[idx_leak - 1], qleak1_0, q0[idx_leak - 1],
              'q', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_q_1'] = Eqn(f'cdm_{pipe_name}_q_1', rhs)

    # i=idx_leak+1
    rhs = cdm(p[idx_leak + 1], p[idx_leak], p0[idx_leak + 1], p0[idx_leak],
              q[idx_leak + 1], qleak2, q0[idx_leak + 1], qleak2_0,
              'p', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_p_2'] = Eqn(f'cdm_{pipe_name}_p_2',
                                           rhs)

    rhs = cdm(p[idx_leak + 1], p[idx_leak], p0[idx_leak + 1], p0[idx_leak],
              q[idx_leak + 1], qleak2, q0[idx_leak + 1], qleak2_0,
              'q', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_q_2'] = Eqn(f'cdm_{pipe_name}_q_2',
                                           rhs)

    # idx_leak+2<=i<=M
    rhs = cdm(p[idx_leak + 2:M + 1], p[idx_leak + 1:M], p0[idx_leak + 2:M + 1], p0[idx_leak + 1:M],
              q[idx_leak + 2:M + 1], q[idx_leak + 1:M], q0[idx_leak + 2:M + 1], q0[idx_leak + 1:M],
              'p', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_p_3'] = Eqn(f'cdm_{pipe_name}_p_3',
                                           rhs)

    rhs = cdm(p[idx_leak + 2:M + 1], p[idx_leak + 1:M], p0[idx_leak + 2:M + 1], p0[idx_leak + 1:M],
              q[idx_leak + 2:M + 1], q[idx_leak + 1:M], q0[idx_leak + 2:M + 1], q0[idx_leak + 1:M],
              'q', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_q_3'] = Eqn(f'cdm_{pipe_name}_q_3',
                                           rhs)

    pb0 = p.value[idx_leak]
    pa_t = [pb0, pb0, pb0, pb0]
    tseries = [0, 300, 300 + 10, 10 * 3600]
    pa = TimeSeriesParam('pa',
                         v_series=pa_t,
                         time_series=tseries)
    artifact['pa'] = pa

    artifact[f'cdm_{pipe_name}_bd1'] = Eqn(f'cdm_{pipe_name}_bd1',
                                           p[idx_leak] - pa)
    artifact[f'cdm_{pipe_name}_bd2'] = Eqn(f'cdm_{pipe_name}_bd2',
                                           qleak1 - qleak2 - q[idx_leak])
    return artifact


def leakage_ngs_pipe_cdm(p: Var,
                         q: Var,
                         lam,
                         va,
                         D,
                         S,
                         dx,
                         dt,
                         M,
                         pipe_name: str,
                         idx_leak,
                         d):
    if not is_integer(M):
        raise TypeError(f'M is {type(M)} instead of integer')

    if not is_number(dx):
        raise TypeError(f'dx is {type(dx)} instead of number')

    artifact = dict()

    qleak1 = Var(f'q_{pipe_name}_leak1', init=q[idx_leak - 1])
    qleak2 = Var(f'q_{pipe_name}_leak2', init=q[idx_leak + 1])
    qleak1_0 = AliasVar(qleak1.name, init=qleak1)
    qleak2_0 = AliasVar(qleak2.name, init=qleak2)
    is_sonic = Param('is_sonic', value=1)

    artifact[qleak1.name] = qleak1
    artifact[qleak2.name] = qleak2
    artifact[qleak1_0.name] = qleak1_0
    artifact[qleak2_0.name] = qleak2_0
    artifact[is_sonic.name] = is_sonic

    p0 = AliasVar(p.name, init=p)
    q0 = AliasVar(q.name, init=q)
    artifact[p0.name] = p0
    artifact[q0.name] = q0

    # 1<=i<=idx_leak-1
    rhs = cdm(p[1:idx_leak], p[0:idx_leak - 1], p0[1:idx_leak], p0[0:idx_leak - 1],
              q[1:idx_leak], q[0:idx_leak - 1], q0[1:idx_leak], q0[0:idx_leak - 1],
              'p', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_p_0'] = Eqn(f'cdm_{pipe_name}_p_0',
                                           rhs)

    rhs = cdm(p[1:idx_leak], p[0:idx_leak - 1], p0[1:idx_leak], p0[0:idx_leak - 1],
              q[1:idx_leak], q[0:idx_leak - 1], q0[1:idx_leak], q0[0:idx_leak - 1],
              'q', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_q_0'] = Eqn(f'cdm_{pipe_name}_q_0',
                                           rhs)

    # i=idx_leak
    rhs = cdm(p[idx_leak], p[idx_leak - 1], p0[idx_leak], p0[idx_leak - 1],
              qleak1, q[idx_leak - 1], qleak1_0, q0[idx_leak - 1],
              'p', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_p_1'] = Eqn(f'cdm_{pipe_name}_p_1', rhs)

    rhs = cdm(p[idx_leak], p[idx_leak - 1], p0[idx_leak], p0[idx_leak - 1],
              qleak1, q[idx_leak - 1], qleak1_0, q0[idx_leak - 1],
              'q', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_q_1'] = Eqn(f'cdm_{pipe_name}_q_1', rhs)

    # i=idx_leak+1
    rhs = cdm(p[idx_leak + 1], p[idx_leak], p0[idx_leak + 1], p0[idx_leak],
              q[idx_leak + 1], qleak2, q0[idx_leak + 1], qleak2_0,
              'p', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_p_2'] = Eqn(f'cdm_{pipe_name}_p_2',
                                           rhs)

    rhs = cdm(p[idx_leak + 1], p[idx_leak], p0[idx_leak + 1], p0[idx_leak],
              q[idx_leak + 1], qleak2, q0[idx_leak + 1], qleak2_0,
              'q', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_q_2'] = Eqn(f'cdm_{pipe_name}_q_2',
                                           rhs)

    # idx_leak+2<=i<=M
    rhs = cdm(p[idx_leak + 2:M + 1], p[idx_leak + 1:M], p0[idx_leak + 2:M + 1], p0[idx_leak + 1:M],
              q[idx_leak + 2:M + 1], q[idx_leak + 1:M], q0[idx_leak + 2:M + 1], q0[idx_leak + 1:M],
              'p', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_p_3'] = Eqn(f'cdm_{pipe_name}_p_3',
                                           rhs)

    rhs = cdm(p[idx_leak + 2:M + 1], p[idx_leak + 1:M], p0[idx_leak + 2:M + 1], p0[idx_leak + 1:M],
              q[idx_leak + 2:M + 1], q[idx_leak + 1:M], q0[idx_leak + 2:M + 1], q0[idx_leak + 1:M],
              'q', lam, va, S, D, dx, dt)
    artifact[f'cdm_{pipe_name}_q_3'] = Eqn(f'cdm_{pipe_name}_q_3',
                                           rhs)

    P2 = p[idx_leak]
    leak_rate = TimeSeriesParam('leak_rate',
                                v_series=[0, 0],
                                time_series=[0, 3 * 3600])
    artifact['leak_rate'] = leak_rate

    C0 = 0.61
    T0 = 293
    Mass = 17.1e-3
    Z = 1
    R = 8.314
    Hcr = 1.3
    d_leak = d
    Ah = np.pi * (d_leak / 2) ** 2
    Pa = 101e3
    P2cr = Pa * (2 / (Hcr + 1)) ** (-Hcr / (Hcr - 1))

    qjleak = leak_rate * (is_sonic *
                          Ah * P2 *
                          (Mass / (Z * R * T0) * Hcr * (2 / (Hcr + 1)) ** ((Hcr + 1) / (Hcr - 1))) ** (1 / 2)
                          + (1 - is_sonic) *
                          C0 * Ah * P2 *
                          (2 * Mass / (Z * R * T0) * Hcr / (Hcr - 1) * (
                                  (Pa / P2) ** (2 / Hcr) - (Pa / P2) ** ((Hcr + 1) / Hcr))) ** (1 / 2))

    artifact[f'cdm_{pipe_name}_bd1'] = Eqn(f'cdm_{pipe_name}_bd1',
                                           q[idx_leak] - qjleak)
    artifact[f'cdm_{pipe_name}_bd2'] = Eqn(f'cdm_{pipe_name}_bd2',
                                           qleak1 - qleak2 - q[idx_leak])
    return artifact
