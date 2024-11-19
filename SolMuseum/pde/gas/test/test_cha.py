import numpy as np
from Solverz import Var, Param, Model, made_numerical, TimeSeriesParam, Rodas, Eqn, Opt, fdae_solver
from SolMuseum.pde import ngs_pipe
import pandas as pd

# %% mdl
L = 51000
p0 = 6621246.69079594
q0 = 14
va = 340
D = 0.5901
S = np.pi * (D / 2) ** 2
lam = 0.03
dx = 500

# def test_cha(datadir):
dt = 1.4706
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
                            method='cha'))

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
fdae, y0 = m1.create_instance()
nfdae, code = made_numerical(fdae, y0, sparse=True, output_code=True)

# %% solution
sol = fdae_solver(nfdae, [0, T], y0, Opt(step_size=dt))


# import matplotlib.pyplot as plt
# plt.plot(sol.T, sol.Y['p'][:, 0])
# plt.plot(sol.T, sol.Y['p'][:, -1])
# plt.show()
#
# plt.plot(sol.T, sol.Y['q'][:, 0])
# plt.plot(sol.T, sol.Y['q'][:, -1])
# plt.show()
#
# res = dict()
# res['pout'] = sol.Y['p'][:, -1]
# res['qin'] = sol.Y['q'][:, 0]
# df = pd.DataFrame(res)
#
# with pd.ExcelWriter(f'res0.xlsx', engine='openpyxl', mode='a') as writer:
#     # Write each DataFrame to a different sheet
#     df.to_excel(writer, sheet_name='cha')

def test_cha(shared_datadir):
    df = pd.read_excel(shared_datadir / 'res0.xlsx',
                       sheet_name='cha',
                       engine='openpyxl',
                       index_col=None
                       )
    qin = np.asarray(df['qin'])
    pout = np.asarray(df['pout'])
    np.testing.assert_allclose(qin, sol.Y['q'][:, 0])
    np.testing.assert_allclose(pout, sol.Y['p'][:, -1])
