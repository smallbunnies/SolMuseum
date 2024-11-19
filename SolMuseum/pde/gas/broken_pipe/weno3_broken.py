from SolMuseum.pde.gas.util import *

__all__ = ['rupture_ngs_pipe_weno3', 'leakage_ngs_pipe_weno3']


def rupture_ngs_pipe_weno3(p: Var,
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
    CUT_DOWNSTREAM = Param('CUT_DOWNSTREAM', 0)

    artifact[qleak1.name] = qleak1
    artifact[qleak2.name] = qleak2
    artifact[CUT_DOWNSTREAM.name] = CUT_DOWNSTREAM

    # index = 0
    artifact['q' + pipe_name + 'bd2'] = Eqn('q' + pipe_name + 'bd2',
                                            S * p[2] - va * q[2] + S * p[0] - va * q[0] - 2 * (
                                                    S * p[1] - va * q[1]))
    # index = 1
    rhs = mol_tvd1_q_eqn_rhs1([p[0], p[1], p[2]],
                              [q[0], q[1], q[2]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_eqn2'] = Ode(f'weno3-q_{pipe_name}_2',
                                              rhs,
                                              q[1])
    rhs = mol_tvd1_p_eqn_rhs1([p[0], p[1], p[2]],
                              [q[0], q[1], q[2]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_eqn2'] = Ode(f'weno3-p_{pipe_name}_2',
                                              rhs,
                                              p[1])
    # 2<= index <= idx_leak-3
    rhs = mol_weno_q_eqn_rhs(p, q, S, va, lam, D, dx, 2, idx_leak - 2)
    artifact['q' + pipe_name + '_leak1'] = Ode(f'weno3-q_{pipe_name}__leak_1',
                                               rhs,
                                               q[2:idx_leak - 2])
    rhs = mol_weno_p_eqn_rhs(p, q, S, va, dx, 2, idx_leak - 2)
    artifact['p' + pipe_name + '_leak1'] = Ode(f'weno3-p_{pipe_name}__leak_1',
                                               rhs,
                                               p[2:idx_leak - 2])

    # index = idx_leak-2
    rhs = mol_weno_q_eqn_rhs1([p[idx_leak - 4], p[idx_leak - 3], p[idx_leak - 2], p[idx_leak - 1], p[idx_leak]],
                              [q[idx_leak - 4], q[idx_leak - 3], q[idx_leak - 2], q[idx_leak - 1], qleak1],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_leak6'] = Ode(f'weno3-q_{pipe_name}__leak_6',
                                               rhs,
                                               q[idx_leak - 2])
    rhs = mol_weno_p_eqn_rhs1([p[idx_leak - 4], p[idx_leak - 3], p[idx_leak - 2], p[idx_leak - 1], p[idx_leak]],
                              [q[idx_leak - 4], q[idx_leak - 3], q[idx_leak - 2], q[idx_leak - 1], qleak1],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_leak6'] = Ode(f'weno3-p_{pipe_name}__leak_6',
                                               rhs,
                                               p[idx_leak - 2])

    # index = idx_leak-1
    rhs = mol_tvd1_q_eqn_rhs1([p[idx_leak - 2], p[idx_leak - 1], p[idx_leak]],
                              [q[idx_leak - 2], q[idx_leak - 1], qleak1],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_leak2'] = Ode(f'weno3-q_{pipe_name}__leak_2',
                                               rhs,
                                               q[idx_leak - 1])
    rhs = mol_tvd1_p_eqn_rhs1([p[idx_leak - 2], p[idx_leak - 1], p[idx_leak]],
                              [q[idx_leak - 2], q[idx_leak - 1], qleak1],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_leak2'] = Ode(f'weno3-p_{pipe_name}__leak_2',
                                               rhs,
                                               p[idx_leak - 1])
    # index = idx_leak
    eqn = S * p[idx_leak] + va * qleak1 + S * p[idx_leak - 2] + va * q[idx_leak - 2] - 2 * (
            S * p[idx_leak - 1] + va * q[idx_leak - 1])
    artifact['q' + pipe_name + '_leak3_1'] = Eqn(f'weno3-q_{pipe_name}__leak_3_1',
                                                 eqn)
    eqn = (S * p[idx_leak + 2] - va * q[idx_leak + 2] + S * p[idx_leak] - va * qleak2
           - 2 * (S * p[idx_leak + 1] - va * q[idx_leak + 1]))
    artifact['q' + pipe_name + '_leak3_2'] = Eqn(f'weno3-q_{pipe_name}__leak_3_2',
                                                 eqn)

    # index = idx_leak+1
    rhs = mol_tvd1_q_eqn_rhs1([p[idx_leak], p[idx_leak + 1], p[idx_leak + 2]],
                              [qleak2, q[idx_leak + 1], q[idx_leak + 2]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_leak4'] = Ode(f'weno3-q_{pipe_name}__leak_4',
                                               rhs,
                                               q[idx_leak + 1])
    rhs = mol_tvd1_p_eqn_rhs1([p[idx_leak], p[idx_leak + 1], p[idx_leak + 2]],
                              [qleak2, q[idx_leak + 1], q[idx_leak + 2]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_leak4'] = Ode(f'weno3-p_{pipe_name}__leak_4',
                                               rhs,
                                               p[idx_leak + 1])

    #  index = idx_leak + 2
    rhs = mol_weno_q_eqn_rhs1([p[idx_leak], p[idx_leak + 1], p[idx_leak + 2], p[idx_leak + 3], p[idx_leak + 4]],
                              [qleak2, q[idx_leak + 1], q[idx_leak + 2], q[idx_leak + 3], q[idx_leak + 4]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_leak7'] = Ode(f'weno3-q_{pipe_name}__leak_7',
                                               rhs,
                                               q[idx_leak + 2])
    rhs = mol_weno_p_eqn_rhs1([p[idx_leak], p[idx_leak + 1], p[idx_leak + 2], p[idx_leak + 3], p[idx_leak + 4]],
                              [qleak2, q[idx_leak + 1], q[idx_leak + 2], q[idx_leak + 3], q[idx_leak + 4]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_leak7'] = Ode(f'weno3-p_{pipe_name}__leak_7',
                                               rhs,
                                               p[idx_leak + 2])

    #  M-2>=index > idx_leak + 2
    rhs = mol_weno_q_eqn_rhs(p, q, S, va, lam, D, dx, idx_leak + 3, M - 1)
    artifact['q' + pipe_name + '_leak5'] = Ode(f'weno3-q_{pipe_name}__leak_5',
                                               rhs,
                                               q[idx_leak + 3: M - 1])
    rhs = mol_weno_p_eqn_rhs(p, q, S, va, dx, idx_leak + 3, M - 1)
    artifact['p' + pipe_name + '_leak5'] = Ode(f'weno3-p_{pipe_name}__leak_5',
                                               rhs,
                                               p[idx_leak + 3:M - 1])
    # index = M-1
    rhs = mol_tvd1_q_eqn_rhs1([p[M - 2], p[M - 1], p[M]],
                              [q[M - 2], q[M - 1], q[M]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_eqn3'] = Ode(f'weno3-q_{pipe_name}_3',
                                              rhs,
                                              q[M - 1])
    rhs = mol_tvd1_p_eqn_rhs1([p[M - 2], p[M - 1], p[M]],
                              [q[M - 2], q[M - 1], q[M]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_eqn3'] = Ode(f'weno3-p_{pipe_name}_3',
                                              rhs,
                                              p[M - 1])

    # index = M
    artifact['p' + pipe_name + 'bd1'] = Eqn('p' + pipe_name + 'bd1',
                                            S * p[M] + va * q[M] + S * p[M - 2] + va * q[
                                                M - 2] - 2 * (S * p[M - 1] + va * q[M - 1]))

    pb0 = p.value[idx_leak]
    pa_t = [pb0, pb0, pb0, pb0]
    tseries = [0, 300, 300 + 10, 10 * 3600]
    pa = TimeSeriesParam('pa',
                         v_series=pa_t,
                         time_series=tseries)
    artifact['pa'] = pa

    artifact[f'_{pipe_name}_bd1'] = Eqn(f'_{pipe_name}_bd1',
                                        p[idx_leak] - pa)
    artifact[f'_{pipe_name}_bd2'] = Eqn(f'_{pipe_name}_bd2',
                                        qleak1 - qleak2 - q[idx_leak])
    return artifact


def leakage_ngs_pipe_weno3(p: Var,
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
    artifact['q' + pipe_name + 'bd2'] = Eqn(q.name + pipe_name + 'bd2',
                                            S * p[2] - va * q[2] + S * p[0] - va * q[0] - 2 * (
                                                    S * p[1] - va * q[1]))
    # index = 1
    rhs = mol_tvd1_q_eqn_rhs1([p[0], p[1], p[2]],
                              [q[0], q[1], q[2]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_eqn2'] = Ode(f'weno3-q_{pipe_name}_2',
                                              rhs,
                                              q[1])
    rhs = mol_tvd1_p_eqn_rhs1([p[0], p[1], p[2]],
                              [q[0], q[1], q[2]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_eqn2'] = Ode(f'weno3-p_{pipe_name}_2',
                                              rhs,
                                              p[1])
    # 2<= index < idx_leak-1
    rhs = mol_weno_q_eqn_rhs(p, q, S, va, lam, D, dx, 2, idx_leak - 1)
    artifact['q' + pipe_name + '_leak1'] = Ode(f'weno3-q_{pipe_name}__leak_1',
                                               rhs,
                                               q[2:idx_leak - 1])
    rhs = mol_weno_p_eqn_rhs(p, q, S, va, dx, 2, idx_leak - 1)
    artifact['p' + pipe_name + '_leak1'] = Ode(f'weno3-p_{pipe_name}__leak_1',
                                               rhs,
                                               p[2:idx_leak - 1])
    # index = idx_leak-1
    rhs = mol_tvd1_q_eqn_rhs1([p[idx_leak - 2], p[idx_leak - 1], p[idx_leak]],
                              [q[idx_leak - 2], q[idx_leak - 1], qleak1],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_leak2'] = Ode(f'weno3-q_{pipe_name}__leak_2',
                                               rhs,
                                               q[idx_leak - 1])
    rhs = mol_tvd1_p_eqn_rhs1([p[idx_leak - 2], p[idx_leak - 1], p[idx_leak]],
                              [q[idx_leak - 2], q[idx_leak - 1], qleak1],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_leak2'] = Ode(f'weno3-p_{pipe_name}__leak_2',
                                               rhs,
                                               p[idx_leak - 1])
    # index = idx_leak
    eqn = S * p[idx_leak] + va * qleak1 + S * p[idx_leak - 2] + va * q[idx_leak - 2] - 2 * (
            S * p[idx_leak - 1] + va * q[idx_leak - 1])
    artifact['q' + pipe_name + '_leak3_1'] = Eqn(f'weno3-q_{pipe_name}__leak_3_1',
                                                 eqn)
    eqn = (S * p[idx_leak + 2] - va * q[idx_leak + 2] + S * p[idx_leak] - va * qleak2
           - 2 * (S * p[idx_leak + 1] - va * q[idx_leak + 1]))
    artifact['q' + pipe_name + '_leak3_2'] = Eqn(f'weno3-q_{pipe_name}__leak_3_2',
                                                 eqn)

    # index = idx_leak+1
    rhs = mol_tvd1_q_eqn_rhs1([p[idx_leak], p[idx_leak + 1], p[idx_leak + 2]],
                              [qleak2, q[idx_leak + 1], q[idx_leak + 2]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_leak4'] = Ode(f'weno3-q_{pipe_name}__leak_4',
                                               rhs,
                                               q[idx_leak + 1])
    rhs = mol_tvd1_p_eqn_rhs1([p[idx_leak], p[idx_leak + 1], p[idx_leak + 2]],
                              [qleak2, q[idx_leak + 1], q[idx_leak + 2]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_leak4'] = Ode(f'weno3-p_{pipe_name}__leak_4',
                                               rhs,
                                               p[idx_leak + 1])
    #  M-2>=index > idx_leak + 1
    rhs = mol_weno_q_eqn_rhs(p, q, S, va, lam, D, dx, idx_leak + 2, M - 1)
    artifact['q' + pipe_name + '_leak5'] = Ode(f'weno3-q_{pipe_name}__leak_5',
                                               rhs,
                                               q[idx_leak + 2: M - 1])
    rhs = mol_weno_p_eqn_rhs(p, q, S, va, dx, idx_leak + 2, M - 1)
    artifact['p' + pipe_name + '_leak5'] = Ode(f'weno3-p_{pipe_name}__leak_5',
                                               rhs,
                                               p[idx_leak + 2:M - 1])
    # index = M-1
    rhs = mol_tvd1_q_eqn_rhs1([p[M - 2], p[M - 1], p[M]],
                              [q[M - 2], q[M - 1], q[M]],
                              S,
                              va,
                              lam,
                              D,
                              dx)
    artifact['q' + pipe_name + '_eqn3'] = Ode(f'weno3-q_{pipe_name}_3',
                                              rhs,
                                              q[M - 1])
    rhs = mol_tvd1_p_eqn_rhs1([p[M - 2], p[M - 1], p[M]],
                              [q[M - 2], q[M - 1], q[M]],
                              S,
                              va,
                              dx)
    artifact['p' + pipe_name + '_eqn3'] = Ode(f'weno3-p_{pipe_name}_3',
                                              rhs,
                                              p[M - 1])

    # index = M
    artifact['p' + pipe_name + 'bd1'] = Eqn(p.name + pipe_name + 'bd1',
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

    artifact[f'_{pipe_name}_bd1'] = Eqn(f'_{pipe_name}_bd1',
                                        q[idx_leak] - qjleak)
    artifact[f'_{pipe_name}_bd2'] = Eqn(f'_{pipe_name}_bd2',
                                        qleak1 - qleak2 - q[idx_leak])
    return artifact
