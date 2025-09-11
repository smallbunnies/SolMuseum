import numpy as np
from Solverz import Var, Param, Model, made_numerical, TimeSeriesParam, Rodas, Eqn, Opt, fdae_solver
from SolMuseum.pde import ngs_pipe
import pandas as pd
import pytest

METHODS = ['weno3', 'cdm', 'cha', 'euler', 'kt1', 'kt2']

@pytest.mark.parametrize('method', METHODS)
def test_normal_pipe(method,
                     shared_datadir,
                     os_name,
                     rtol,
                     atol):

    # %% mdl
    L = 51000
    p0 = 6621246.69079594
    q0 = 14
    va = 340
    D = 0.5901
    S = np.pi * (D / 2) ** 2
    lam = 0.03
    dx = 500

    if method == 'cha':
        dt = 1.4706
    else:
        dt = 60

    M = int(L / dx)
    m1 = Model()
    m1.p = Var('p', value=p0 * np.ones((M + 1,)))
    m1.q = Var('q', value=q0 * np.ones((M + 1,)))

    m1.__dict__.update(ngs_pipe(m1.p,
                                m1.q,
                                lam,
                                va,
                                D,
                                S,
                                dx,
                                dt,
                                M,
                                '1',
                                method=method))

    T = 5 * 3600
    pb1 = 1e6
    pb0 = 6621246.69079594
    pb_t = [pb0, pb0, pb1, pb1]
    tseries = [0, 1000, 1000 + 10 * dt, T]
    m1.pb = TimeSeriesParam('pb',
                            v_series=pb_t,
                            time_series=tseries)
    m1.qb = Param('qb', q0)
    m1.bd1 = Eqn('bd1', m1.p[0] - m1.pb)
    m1.bd2 = Eqn('bd2', m1.q[M] - m1.qb)

    if method in ['cdm', 'euler', 'cha']:
        fdae, y0 = m1.create_instance()
        nfdae, code = made_numerical(fdae, y0, sparse=True, output_code=True)
        sol = fdae_solver(nfdae, [0, T], y0, Opt(step_size=dt))
    elif method in ['kt1', 'kt2', 'weno3']:
        dae, y0 = m1.create_instance()
        ndae, code = made_numerical(dae, y0, sparse=True, output_code=True)
        sol = Rodas(ndae,
                    np.linspace(0, T, 301),
                    y0)

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
