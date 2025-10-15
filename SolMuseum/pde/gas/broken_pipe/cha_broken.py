from SolMuseum.pde.gas.util import *

__all__ = ['rupture_ngs_pipe_cha', 'leakage_ngs_pipe_cha']


def rupture_ngs_pipe_cha(p: Var,
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

    rhs = cha(p[1:idx_leak],
              q[1:idx_leak],
              p0[0:idx_leak - 1],
              q0[0:idx_leak - 1],
              '+',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_pos0'] = Eqn(f'cha_{pipe_name}_pos0',
                                            rhs)

    rhs = cha(p[idx_leak],
              qleak1,
              p0[idx_leak - 1],
              q0[idx_leak - 1],
              '+',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_pos1'] = Eqn(f'cha_{pipe_name}_pos1',
                                            rhs)

    rhs = cha(p[idx_leak + 2:M + 1],
              q[idx_leak + 2:M + 1],
              p0[idx_leak + 1:M],
              q0[idx_leak + 1:M],
              '+',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_pos2'] = Eqn(f'cha_{pipe_name}_pos2', rhs)

    rhs = cha(p[idx_leak + 1],
              q[idx_leak + 1],
              p0[idx_leak],
              qleak2_0,
              '+',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_pos3'] = Eqn(f'cha_{pipe_name}_pos3', rhs)

    rhs = cha(p[0:idx_leak - 1],
              q[0:idx_leak - 1],
              p0[1:idx_leak],
              q0[1:idx_leak],
              '-',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_neg0'] = Eqn(f'cha_{pipe_name}_neg0',
                                            rhs)

    rhs = cha(p[idx_leak - 1],
              q[idx_leak - 1],
              p0[idx_leak],
              qleak1_0,
              '-',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_neg1'] = Eqn(f'cha_{pipe_name}_neg1',
                                            rhs)

    rhs = cha(p[idx_leak + 1:M],
              q[idx_leak + 1:M],
              p0[idx_leak + 2:M + 1],
              q0[idx_leak + 2:M + 1],
              '-',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_neg2'] = Eqn(f'cha_{pipe_name}_neg2',
                                            rhs)

    rhs = cha(p[idx_leak],
              qleak2,
              p0[idx_leak + 1],
              q0[idx_leak + 1],
              '-',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_neg3'] = Eqn(f'cha_{pipe_name}_neg3',
                                            rhs)

    pb0 = p.value[idx_leak]
    pa_t = [pb0, pb0, pb0, pb0]
    tseries = [0, 300, 300 + 10, 10 * 3600]
    pa = TimeSeriesParam('pa',
                         v_series=pa_t,
                         time_series=tseries)
    artifact['pa'] = pa

    artifact[f'cha_{pipe_name}_bd1'] = Eqn(f'cha_{pipe_name}_bd1',
                                           p[idx_leak] - pa)
    artifact[f'cha_{pipe_name}_bd2'] = Eqn(f'cha_{pipe_name}_bd2',
                                           qleak1 - qleak2 - q[idx_leak])
    return artifact


def leakage_ngs_pipe_cha(p: Var,
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

    rhs = cha(p[1:idx_leak],
              q[1:idx_leak],
              p0[0:idx_leak - 1],
              q0[0:idx_leak - 1],
              '+',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_pos0'] = Eqn(f'cha_{pipe_name}_pos0',
                                            rhs)

    rhs = cha(p[idx_leak],
              qleak1,
              p0[idx_leak - 1],
              q0[idx_leak - 1],
              '+',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_pos1'] = Eqn(f'cha_{pipe_name}_pos1',
                                            rhs)

    rhs = cha(p[idx_leak + 2:M + 1],
              q[idx_leak + 2:M + 1],
              p0[idx_leak + 1:M],
              q0[idx_leak + 1:M],
              '+',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_pos2'] = Eqn(f'cha_{pipe_name}_pos2', rhs)

    rhs = cha(p[idx_leak + 1],
              q[idx_leak + 1],
              p0[idx_leak],
              qleak2_0,
              '+',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_pos3'] = Eqn(f'cha_{pipe_name}_pos3', rhs)

    rhs = cha(p[0:idx_leak - 1],
              q[0:idx_leak - 1],
              p0[1:idx_leak],
              q0[1:idx_leak],
              '-',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_neg0'] = Eqn(f'cha_{pipe_name}_neg0',
                                            rhs)

    rhs = cha(p[idx_leak - 1],
              q[idx_leak - 1],
              p0[idx_leak],
              qleak1_0,
              '-',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_neg1'] = Eqn(f'cha_{pipe_name}_neg1',
                                            rhs)

    rhs = cha(p[idx_leak + 1:M],
              q[idx_leak + 1:M],
              p0[idx_leak + 2:M + 1],
              q0[idx_leak + 2:M + 1],
              '-',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_neg2'] = Eqn(f'cha_{pipe_name}_neg2',
                                            rhs)

    rhs = cha(p[idx_leak],
              qleak2,
              p0[idx_leak + 1],
              q0[idx_leak + 1],
              '-',
              lam,
              va,
              S,
              D,
              dx)
    artifact[f'cha_{pipe_name}_neg3'] = Eqn(f'cha_{pipe_name}_neg3',
                                            rhs)

    P2 = p[idx_leak]
    leak_rate = TimeSeriesParam('leak_rate_' + pipe_name,
                                v_series=[0, 0],
                                time_series=[0, 3 * 3600])
    artifact[leak_rate.name] = leak_rate

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

    qleak = leak_rate * (is_sonic *
                         Ah * P2 *
                         (Mass / (Z * R * T0) * Hcr * (2 / (Hcr + 1)) ** ((Hcr + 1) / (Hcr - 1))) ** (1 / 2)
                         + (1 - is_sonic) *
                         C0 * Ah * P2 *
                         (2 * Mass / (Z * R * T0) * Hcr / (Hcr - 1) * (
                                 (Pa / P2) ** (2 / Hcr) - (Pa / P2) ** ((Hcr + 1) / Hcr))) ** (1 / 2))

    artifact[f'cha_{pipe_name}_bd1'] = Eqn(f'cha_{pipe_name}_bd1',
                                           q[idx_leak] - qleak)
    artifact[f'cha_{pipe_name}_bd2'] = Eqn(f'cha_{pipe_name}_bd2',
                                           qleak1 - qleak2 - q[idx_leak])
    return artifact
