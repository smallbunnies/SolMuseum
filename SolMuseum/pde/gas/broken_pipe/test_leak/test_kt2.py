import numpy as np
from Solverz import Model, Param, Var, made_numerical, Rodas, Opt, Eqn, TimeSeriesParam
import pandas as pd
from SolMuseum.pde.gas.broken_pipe import leakage_pipe


# %%
def test_kt2(shared_datadir):
    # %%
    L = 51000
    T = 2000
    lam = 0.03
    dx = 500
    M = int(L / dx)
    idx_leak = int(M * 0.5)
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
                                   0,
                                   M,
                                   '1',
                                   idx_leak,
                                   0.9 * m.D,
                                   method='kt2'))
    m.theta = Param('theta', 2)
    m.pb = Param('pb', p0)
    m.qb = Param('qb', q0)
    m.bd1 = Eqn('bd1', m.p[0] - m.pb)
    m.bd2 = Eqn('bd2', m.q[M] - m.qb)

    # %% solution
    sdae, y0 = m.create_instance()
    ndae, code = made_numerical(sdae, y0, output_code=True, sparse=True)

    ndae.p['leak_rate'] = TimeSeriesParam('leak_rate',
                                          v_series=[0, 0, 1, 1],
                                          time_series=[0, 100, 105, 3600])

    sol = Rodas(ndae, np.linspace(0, T, 401), y0, Opt(pbar=True))

    # %%
    # import matplotlib.pyplot as plt
    #
    # plt.plot(sol.T, sol.Y['p'][:, 0])
    # plt.plot(sol.T, sol.Y['p'][:, -1])
    # plt.plot(sol.T, sol.Y['p'][:, idx_leak])
    # plt.plot(sol.T, sol.Y['p'][:, idx_leak - 1])
    # plt.show()
    #
    # plt.plot(sol.T, sol.Y['q'][:, 0])
    # # plt.plot(sol.T, sol.Y['q_1_leak1'])
    # plt.plot(sol.T, sol.Y['q'][:, -1])
    # plt.show()
    #
    # res = dict()
    # res['pout'] = sol.Y['p'][:, -1]
    # res['qin'] = sol.Y['q'][:, 0]
    # res['qupstream'] = sol.Y['q_1_leak1'].reshape(-1)
    # df = pd.DataFrame(res)
    #
    # with pd.ExcelWriter(f'data/res.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    #     # Write each DataFrame to a different sheet
    #     df.to_excel(writer, sheet_name='kt2')

    df = pd.read_excel(shared_datadir / 'res.xlsx',
                       sheet_name='kt2',
                       engine='openpyxl',
                       index_col=None
                       )
    qin = np.asarray(df['qin'])
    pout = np.asarray(df['pout'])
    qupstream = np.asarray(df['qupstream'])
    np.testing.assert_allclose(qin, sol.Y['q'][:, 0])
    np.testing.assert_allclose(pout, sol.Y['p'][:, -1])
    np.testing.assert_allclose(qupstream, sol.Y['q_1_leak1'].reshape(-1))
