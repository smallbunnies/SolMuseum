from SolMuseum.pde.gas.util import *

__all__ = ['rupture_ngs_pipe_kt2', 'leakage_ngs_pipe_kt2']


def rupture_ngs_pipe_kt2(p: Var,
                         q: Var,
                         lam,
                         va,
                         D,
                         S,
                         dx,
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

    artifact[qleak1.name] = qleak1
    artifact[qleak2.name] = qleak2

    # index = 0
    artifact['q' + pipe_name + 'bd2'] = Eqn(q.name + 'bd2',
                                            S * p[2] - va * q[2] + S * p[0] - va * q[0] - 2 * (
                                                    S * p[1] - va * q[1]))
    # index = 1
    rhs = mol_tvd1_q_eqn_rhs0([p[0], p[1], p[2]],
                              [q[0], q[1], q[2]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q_' + pipe_name + '_eqn1'] = Ode(f'kt2-q{pipe_name}_1',
                                               rhs,
                                               q[1])

    rhs = mol_tvd1_p_eqn_rhs0([p[0], p[1], p[2]],
                              [q[0], q[1], q[2]],
                              S,
                              va,
                              dx)
    artifact['p_' + pipe_name + '_eqn1'] = Ode(f'kt2-p{pipe_name}_1',
                                               rhs,
                                               p[1])

    # 2<= index <= idx_leak-3
    rhs = mol_tvd1_q_eqn_rhs0([p[0:idx_leak - 2], p[1:idx_leak - 1], p[2:idx_leak]],
                              [q[0:idx_leak - 2], q[1:idx_leak - 1], q[2:idx_leak]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_eqn1'] = Ode(f'kt1-q{pipe_name}_1',
                                              rhs,
                                              q[1:idx_leak - 1])
    rhs = mol_tvd1_p_eqn_rhs0([p[0:idx_leak - 2], p[1:idx_leak - 1], p[2:idx_leak]],
                              [q[0:idx_leak - 2], q[1:idx_leak - 1], q[2:idx_leak]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_eqn1'] = Ode(f'kt1-p{pipe_name}_1',
                                              rhs,
                                              p[1:idx_leak - 1])

    # index = idx_leak-1
    rhs = mol_tvd1_q_eqn_rhs0([p[idx_leak - 2], p[idx_leak - 1], p[idx_leak]],
                              [q[idx_leak - 2], q[idx_leak - 1], qleak1],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_leak1'] = Ode(f'kt1-q{pipe_name}_leak_1',
                                               rhs,
                                               q[idx_leak - 1])
    rhs = mol_tvd1_p_eqn_rhs0([p[idx_leak - 2], p[idx_leak - 1], p[idx_leak]],
                              [q[idx_leak - 2], q[idx_leak - 1], qleak1],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_leak1'] = Ode(f'kt1-p{pipe_name}_leak_1',
                                               rhs,
                                               p[idx_leak - 1])
    # index = idx_leak
    eqn = S * p[idx_leak] + va * qleak1 + S * p[idx_leak - 2] + va * q[idx_leak - 2] - 2 * (
            S * p[idx_leak - 1] + va * q[idx_leak - 1])
    artifact['q' + pipe_name + '_leak3_1'] = Eqn(f'kt1-q{pipe_name}_leak_3_1',
                                                 eqn)
    eqn = (S * p[idx_leak + 2] - va * q[idx_leak + 2] + S * p[idx_leak] - va * qleak2
           - 2 * (S * p[idx_leak + 1] - va * q[idx_leak + 1]))
    artifact['q' + pipe_name + '_leak3_2'] = Eqn(f'kt1-q{pipe_name}_leak_3_2',
                                                 eqn)
    # index = idx_leak+1
    rhs = mol_tvd1_q_eqn_rhs0([p[idx_leak], p[idx_leak + 1], p[idx_leak + 2]],
                              [qleak2, q[idx_leak + 1], q[idx_leak + 2]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_leak4'] = Ode(f'kt1-q{pipe_name}_leak_4',
                                               rhs,
                                               q[idx_leak + 1])
    rhs = mol_tvd1_p_eqn_rhs0([p[idx_leak], p[idx_leak + 1], p[idx_leak + 2]],
                              [qleak2, q[idx_leak + 1], q[idx_leak + 2]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_leak4'] = Ode(f'kt1-p{pipe_name}_leak_4',
                                               rhs,
                                               p[idx_leak + 1])
    #  M-1>=index >= idx_leak + 2
    rhs = mol_tvd1_q_eqn_rhs0([p[idx_leak + 1:M - 1], p[idx_leak + 2:M], p[idx_leak + 3:M + 1]],
                              [q[idx_leak + 1:M - 1], q[idx_leak + 2:M], q[idx_leak + 3:M + 1]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_eqn3'] = Ode(f'kt1-q{pipe_name}_3',
                                              rhs,
                                              q[idx_leak + 2:M])
    rhs = mol_tvd1_p_eqn_rhs0([p[idx_leak + 1:M - 1], p[idx_leak + 2:M], p[idx_leak + 3:M + 1]],
                              [q[idx_leak + 1:M - 1], q[idx_leak + 2:M], q[idx_leak + 3:M + 1]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_eqn3'] = Ode(f'kt1-p{pipe_name}_3',
                                              rhs,
                                              p[idx_leak + 2:M])
    # index = M
    artifact['p' + pipe_name + 'bd1'] = Eqn(p.name + 'bd1',
                                            S * p[M] + va * q[M] + S * p[M - 2] + va * q[
                                                M - 2] - 2 * (S * p[M - 1] + va * q[M - 1]))

    pb0 = p.value[idx_leak]
    pa_t = [pb0, pb0, pb0, pb0]
    tseries = [0, 300, 300 + 10, 10 * 3600]
    pa = TimeSeriesParam('pa',
                         v_series=pa_t,
                         time_series=tseries)
    artifact['pa'] = pa

    artifact[f'kt1_{pipe_name}_bd1'] = Eqn(f'kt1_{pipe_name}_bd1',
                                           p[idx_leak] - pa)
    artifact[f'kt1_{pipe_name}_bd2'] = Eqn(f'kt1_{pipe_name}_bd2',
                                           qleak1 - qleak2 - q[idx_leak])
    return artifact


def leakage_ngs_pipe_kt2(p: Var,
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

    artifact[qleak1.name] = qleak1
    artifact[qleak2.name] = qleak2

    is_sonic = Param('is_sonic', value=1)
    artifact[is_sonic.name] = is_sonic

    # index = 0
    artifact['q' + pipe_name + 'bd2'] = Eqn(q.name + 'bd2',
                                            S * p[2] - va * q[2] + S * p[0] - va * q[0] - 2 * (
                                                    S * p[1] - va * q[1]))
    # 1<= index <= idx_leak-2
    rhs = mol_tvd1_q_eqn_rhs0([p[0:idx_leak - 2], p[1:idx_leak - 1], p[2:idx_leak]],
                              [q[0:idx_leak - 2], q[1:idx_leak - 1], q[2:idx_leak]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_eqn1'] = Ode(f'kt1-q{pipe_name}_1',
                                              rhs,
                                              q[1:idx_leak - 1])
    rhs = mol_tvd1_p_eqn_rhs0([p[0:idx_leak - 2], p[1:idx_leak - 1], p[2:idx_leak]],
                              [q[0:idx_leak - 2], q[1:idx_leak - 1], q[2:idx_leak]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_eqn1'] = Ode(f'kt1-p{pipe_name}_1',
                                              rhs,
                                              p[1:idx_leak - 1])

    # index = idx_leak-1
    rhs = mol_tvd1_q_eqn_rhs0([p[idx_leak - 2], p[idx_leak - 1], p[idx_leak]],
                              [q[idx_leak - 2], q[idx_leak - 1], qleak1],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_leak1'] = Ode(f'kt1-q{pipe_name}_leak_1',
                                               rhs,
                                               q[idx_leak - 1])
    rhs = mol_tvd1_p_eqn_rhs0([p[idx_leak - 2], p[idx_leak - 1], p[idx_leak]],
                              [q[idx_leak - 2], q[idx_leak - 1], qleak1],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_leak1'] = Ode(f'kt1-p{pipe_name}_leak_1',
                                               rhs,
                                               p[idx_leak - 1])
    # index = idx_leak
    eqn = S * p[idx_leak] + va * qleak1 + S * p[idx_leak - 2] + va * q[idx_leak - 2] - 2 * (
            S * p[idx_leak - 1] + va * q[idx_leak - 1])
    artifact['q' + pipe_name + '_leak3_1'] = Eqn(f'kt1-q{pipe_name}_leak_3_1',
                                                 eqn)
    eqn = (S * p[idx_leak + 2] - va * q[idx_leak + 2] + S * p[idx_leak] - va * qleak2
           - 2 * (S * p[idx_leak + 1] - va * q[idx_leak + 1]))
    artifact['q' + pipe_name + '_leak3_2'] = Eqn(f'kt1-q{pipe_name}_leak_3_2',
                                                 eqn)
    # index = idx_leak+1
    rhs = mol_tvd1_q_eqn_rhs0([p[idx_leak], p[idx_leak + 1], p[idx_leak + 2]],
                              [qleak2, q[idx_leak + 1], q[idx_leak + 2]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_leak4'] = Ode(f'kt1-q{pipe_name}_leak_4',
                                               rhs,
                                               q[idx_leak + 1])
    rhs = mol_tvd1_p_eqn_rhs0([p[idx_leak], p[idx_leak + 1], p[idx_leak + 2]],
                              [qleak2, q[idx_leak + 1], q[idx_leak + 2]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_leak4'] = Ode(f'kt1-p{pipe_name}_leak_4',
                                               rhs,
                                               p[idx_leak + 1])
    #  M-1>=index >= idx_leak + 2
    rhs = mol_tvd1_q_eqn_rhs0([p[idx_leak + 1:M - 1], p[idx_leak + 2:M], p[idx_leak + 3:M + 1]],
                              [q[idx_leak + 1:M - 1], q[idx_leak + 2:M], q[idx_leak + 3:M + 1]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_eqn3'] = Ode(f'kt1-q{pipe_name}_3',
                                              rhs,
                                              q[idx_leak + 2:M])
    rhs = mol_tvd1_p_eqn_rhs0([p[idx_leak + 1:M - 1], p[idx_leak + 2:M], p[idx_leak + 3:M + 1]],
                              [q[idx_leak + 1:M - 1], q[idx_leak + 2:M], q[idx_leak + 3:M + 1]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_eqn3'] = Ode(f'kt1-p{pipe_name}_3',
                                              rhs,
                                              p[idx_leak + 2:M])
    # index = M
    artifact['p' + pipe_name + 'bd1'] = Eqn(p.name + 'bd1',
                                            S * p[M] + va * q[M] + S * p[M - 2] + va * q[
                                                M - 2] - 2 * (S * p[M - 1] + va * q[M - 1]))

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

    artifact[f'kt1_{pipe_name}_bd1'] = Eqn(f'kt1_{pipe_name}_bd1',
                                           q[idx_leak] - qjleak)
    artifact[f'kt1_{pipe_name}_bd2'] = Eqn(f'kt1_{pipe_name}_bd2',
                                           qleak1 - qleak2 - q[idx_leak])
    return artifact
