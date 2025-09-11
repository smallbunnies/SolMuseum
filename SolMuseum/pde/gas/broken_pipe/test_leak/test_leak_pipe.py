import numpy as np
from Solverz import Model, Param, Var, made_numerical, fdae_solver, Opt, Eqn, TimeSeriesParam, Rodas
import pandas as pd
from SolMuseum.pde.gas.broken_pipe import leakage_pipe

import pytest

METHODS = ['weno3', 'cdm', 'cha', 'euler', 'kt1', 'kt2']

# %%
@pytest.mark.parametrize('method', METHODS)
def test_leak_pipe(method,
                   shared_datadir,
                   os_name,
                   rtol,
                   atol):
    # %%
    L = 51000
    T = 2000
    lam = 0.03
    if method != 'cha':
        dx = 500
    else:
        dx = 100
    M = int(L / dx)
    idx_leak = int(M * 0.5)
    if method != 'cha':
        dt = 5
    else:
        dt = 0.1471 * 2
    p0 = 6621246.69079594
    q0 = 14

    m = Model()
    m.D = Param('D', value=0.5901)
    S = np.pi * (m.D / 2) ** 2
    m.lam = Param('lam', lam)
    m.va = Param('va', value=340)
    m.p = Var('p', value=p0 * np.ones((M + 1)))
    m.q = Var('q', value=q0 * np.ones((M + 1)))

    m.__dict__.update(leakage_pipe(m.p,
                                   m.q,
                                   m.lam,
                                   m.va,
                                   m.D,
                                   S,
                                   dx,
                                   dt,
                                   M,
                                   '1',
                                   idx_leak,
                                   method=method,
                                   d=0.9 * m.D))
    m.theta = Param('theta', value=2.0)
    m.pb = Param('pb', p0)
    m.qb = Param('qb', q0)
    m.bd1 = Eqn('bd1', m.p[0] - m.pb)
    m.bd2 = Eqn('bd2', m.q[M] - m.qb)

    # %% solution
    smdl, y0 = m.create_instance()
    nmdl, code = made_numerical(smdl, y0, output_code=True, sparse=True)

    nmdl.p['leak_rate'] = TimeSeriesParam('leak_rate',
                                          v_series=[0, 0, 1, 1],
                                          time_series=[0, 100, 105, 3600])

    if method in ['cha', 'cdm', 'euler']:
        sol = fdae_solver(nmdl, [0, T], y0, Opt(step_size=dt))
    elif method in ['kt1', 'kt2', 'weno3']:
        sol = Rodas(nmdl, np.linspace(0, T, 401), y0, Opt(pbar=True))

    df = pd.read_excel(shared_datadir / f'res_{os_name}.xlsx',
                       sheet_name=method,
                       engine='openpyxl',
                       index_col=None
                       )
    qin = np.asarray(df['qin'])
    pout = np.asarray(df['pout'])
    try:
        np.testing.assert_allclose(sol.Y['q'][:, 0], qin, rtol=rtol, atol=atol)
    except AssertionError:
        diff = sol.Y['q'][:, 0] - qin
        assert np.mean(np.abs(diff)) <= rtol * 1e-2
    try:
        np.testing.assert_allclose(sol.Y['p'][:, -1], pout, rtol=rtol, atol=atol)
    except AssertionError:
        diff = sol.Y['p'][:, -1] - pout
        assert np.mean(np.abs(diff)) <= rtol * 1e-2

    # if method in ['cdm', 'kt2', 'kt1']:
    #     qupstream = np.asarray(df['qupstream'])
    #     np.testing.assert_allclose(sol.Y['q_1_leak1'].reshape(-1), qupstream, rtol=rtol, atol=atol)
