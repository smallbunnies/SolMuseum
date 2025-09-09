import numpy as np
from Solverz import Model, Param, Var, made_numerical, fdae_solver, Opt, Eqn, TimeSeriesParam, Rodas
import pandas as pd
from SolMuseum.pde.gas.broken_pipe import rupture_pipe

import pytest

METHODS = ['weno3', 'cdm', 'cha', 'euler', 'kt1', 'kt2']

@pytest.mark.parametrize('method', METHODS)
def test_rup_pipe(method,
                  shared_datadir,
                  rtol,
                  atol):
    # %%
    L = 51000
    if method == 'kt2':
        T = 305
    else:
        T = 2000
    lam = 0.03
    if method == 'cha':
        dx = 50
    else:
        dx = 100
    M = int(L / dx)
    idx_leak = int(M * 0.5)
    if method != 'cha':
        dt = 10
    else:
        dt = 0.1471
    p0 = 6621246.69079594
    q0 = 14

    m = Model()
    m.D = Param('D', value=0.5901)
    S = np.pi * (m.D / 2) ** 2
    m.lam = Param('lam', lam)
    m.va = Param('va', value=340)
    m.p = Var('p', value=p0 * np.ones((M + 1)))
    m.q = Var('q', value=q0 * np.ones((M + 1)))

    m.__dict__.update(rupture_pipe(m.p,
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
                                   method=method))
    if method == 'kt2':
        m.theta = Param('theta', value=2.0)
    m.pb = Param('pb', p0)
    m.qb = Param('qb', q0)
    m.bd1 = Eqn('bd1', m.p[0] - m.pb)
    m.bd2 = Eqn('bd2', m.q[M] - m.qb)

    # %% solution
    smdl, y0 = m.create_instance()
    nmdl, code = made_numerical(smdl, y0, output_code=True, sparse=True)

    pbar = 0.101325e6
    pb0 = m.p.value[idx_leak]
    pa_t = [pb0, pb0, pbar, pbar]
    tseries = [0, 300, 300 + 10, 10 * 3600]
    nmdl.p['pa'] = TimeSeriesParam('pa',
                                   v_series=pa_t,
                                   time_series=tseries)

    if method in ['cha', 'cdm', 'euler']:
        sol = fdae_solver(nmdl, [0, T], y0, Opt(step_size=dt))
    elif method in ['kt1', 'kt2', 'weno3']:
        sol = Rodas(nmdl, np.linspace(0, T, 201), y0)

    # %%
    df = pd.read_excel(shared_datadir / 'res.xlsx',
                       sheet_name=method,
                       engine='openpyxl',
                       index_col=None
                       )
    qin = np.asarray(df['qin'])
    pout = np.asarray(df['pout'])
    np.testing.assert_allclose(qin, sol.Y['q'][:, 0], rtol=rtol, atol=atol)
    np.testing.assert_allclose(pout, sol.Y['p'][:, -1], rtol=rtol, atol=atol)
    if method in ['cdm', 'kt1', 'kt2', 'weno3']:
        qupstream = np.asarray(df['qupstream'])
        np.testing.assert_allclose(qupstream, sol.Y['q_1_leak1'].reshape(-1), rtol=rtol, atol=atol)
    if method in ['euler']:
        pupstream = np.asarray(df['pupstream'])
        np.testing.assert_allclose(pupstream, sol.Y['p'][:, idx_leak - 1], rtol=rtol, atol=atol)
