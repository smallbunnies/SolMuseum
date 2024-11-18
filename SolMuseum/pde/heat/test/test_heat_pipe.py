import numpy as np
from Solverz import Var, Param, Model, made_numerical, TimeSeriesParam, Rodas, Eqn, Opt, fdae_solver
from SolMuseum.pde import heat_pipe
import pandas as pd


def test_heat_pipe_iu(datadir):
    # modelling
    L = 9250
    dx = 370
    M = int(L / dx)
    Tinitial = np.zeros(M + 1)
    lam = 1 / 0.35
    Cp = 4182
    Ta = -10
    D = 1.4
    S = np.pi * (D / 2) ** 2
    rho = 960
    Tamb = -10

    for i in range(0, len(Tinitial)):
        fai = np.exp(-lam * i * dx / (Cp * np.abs(2543.5)))
        Tinitial[i] = 90.1724637976347 * fai + Ta * (1 - fai)

    df = pd.read_excel(datadir / 'param.xlsx',
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )
    mvariation = np.asarray(df['Sheet1']['m(m3/s)']) * rho
    Tmeasured = np.asarray(df['Sheet1']['Toutlet'])
    m = Model()
    m.T = Var('T', Tinitial)
    m.m = TimeSeriesParam('m',
                          mvariation,
                          np.linspace(0, 20 * 3600, 401))
    m.lam = Param('lam', lam)
    m.rho = Param('rho', rho)
    m.S = Param('S', S)
    m.Tamb = Param('Tamb', Tamb)
    m.Cp = Param('Cp', Cp)
    m.dt = Param('dt', 180)
    m.__dict__.update(heat_pipe(m.T,
                                m.m,
                                m.lam,
                                m.rho,
                                m.Cp,
                                m.S,
                                m.Tamb,
                                dx,
                                m.dt,
                                M,
                                '1',
                                method='iu'))
    m.Tsource = TimeSeriesParam('Tsource',
                                np.asarray(df['Sheet1']['Tsource']),
                                np.linspace(0, 20 * 3600, 401))
    m.Bd = Eqn('Bd', m.Tsource - m.T[0])
    heatpipe, y0 = m.create_instance()
    fdae = made_numerical(heatpipe, y0, sparse=True)

    sol = fdae_solver(fdae, [0, 20 * 3600], y0, Opt(step_size=180))

    res = pd.read_excel(datadir / 'res.xlsx',
                        sheet_name='Tout',
                        engine='openpyxl',
                        index_col=None
                        )
    bench_iu = res['iu']

    rmse = np.mean(np.abs(Tmeasured - sol.Y['T'][:, -1]) ** 2) ** (1 / 2)
    assert rmse < 0.2534
    np.testing.assert_allclose(bench_iu, sol.Y['T'][:, -1])


def test_heat_pipe_yao(datadir):
    # modelling
    L = 9250
    dx = 370
    M = int(L / dx)
    Tinitial = np.zeros(M + 1)
    lam = 1 / 0.35
    Cp = 4182
    Ta = -10
    D = 1.4
    S = np.pi * (D / 2) ** 2
    rho = 960
    Tamb = -10

    for i in range(0, len(Tinitial)):
        fai = np.exp(-lam * i * dx / (Cp * np.abs(2543.5)))
        Tinitial[i] = 90.1724637976347 * fai + Ta * (1 - fai)

    df = pd.read_excel(datadir / 'param.xlsx',
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )
    mvariation = np.asarray(df['Sheet1']['m(m3/s)']) * rho
    Tmeasured = np.asarray(df['Sheet1']['Toutlet'])
    m = Model()
    m.T = Var('T', Tinitial)
    m.m = TimeSeriesParam('m',
                          mvariation,
                          np.linspace(0, 20 * 3600, 401))
    m.lam = Param('lam', lam)
    m.rho = Param('rho', rho)
    m.S = Param('S', S)
    m.Tamb = Param('Tamb', Tamb)
    m.Cp = Param('Cp', Cp)
    m.dt = Param('dt', 180)
    m.__dict__.update(heat_pipe(m.T,
                                m.m,
                                m.lam,
                                m.rho,
                                m.Cp,
                                m.S,
                                m.Tamb,
                                dx,
                                m.dt,
                                M,
                                '1',
                                method='yao'))
    m.Tsource = TimeSeriesParam('Tsource',
                                np.asarray(df['Sheet1']['Tsource']),
                                np.linspace(0, 20 * 3600, 401))
    m.Bd = Eqn('Bd', m.Tsource - m.T[0])
    heatpipe, y0 = m.create_instance()
    fdae = made_numerical(heatpipe, y0, sparse=True)

    sol = fdae_solver(fdae, [0, 20 * 3600], y0, Opt(step_size=180))

    res = pd.read_excel(datadir / 'res.xlsx',
                        sheet_name='Tout',
                        engine='openpyxl',
                        index_col=None
                        )

    bench = res['yao']

    rmse = np.mean(np.abs(Tmeasured - sol.Y['T'][:, -1]) ** 2) ** (1 / 2)
    assert rmse < 0.1515
    np.testing.assert_allclose(bench, sol.Y['T'][:, -1])


def test_heat_pipe_rodaskt2(datadir):
    # modelling
    L = 9250
    dx = 370
    M = int(L / dx)
    Tinitial = np.zeros(M + 1)
    lam = 1 / 0.35
    Cp = 4182
    Ta = -10
    D = 1.4
    S = np.pi * (D / 2) ** 2
    rho = 960
    Tamb = -10

    for i in range(0, len(Tinitial)):
        fai = np.exp(-lam * i * dx / (Cp * np.abs(2543.5)))
        Tinitial[i] = 90.1724637976347 * fai + Ta * (1 - fai)

    df = pd.read_excel(datadir / 'param.xlsx',
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )
    mvariation = np.asarray(df['Sheet1']['m(m3/s)']) * rho
    Tmeasured = np.asarray(df['Sheet1']['Toutlet'])
    m = Model()
    m.T = Var('T', Tinitial)
    m.m = TimeSeriesParam('m',
                          mvariation,
                          np.linspace(0, 20 * 3600, 401))
    m.lam = Param('lam', lam)
    m.rho = Param('rho', rho)
    m.S = Param('S', S)
    m.Tamb = Param('Tamb', Tamb)
    m.Cp = Param('Cp', Cp)
    m.__dict__.update(heat_pipe(m.T,
                                m.m,
                                m.lam,
                                m.rho,
                                m.Cp,
                                m.S,
                                m.Tamb,
                                dx,
                                0,
                                M,
                                '1'))
    m.Tsource = TimeSeriesParam('Tsource',
                                np.asarray(df['Sheet1']['Tsource']),
                                np.linspace(0, 20 * 3600, 401))
    m.Bd = Eqn('Bd', m.Tsource - m.T[0])
    heatpipe, y0 = m.create_instance()
    dae = made_numerical(heatpipe, y0, sparse=True)

    dae.p['theta'] = np.array([2])
    soltheta2 = Rodas(dae, [0, 20 * 3600], y0, Opt(fix_h=True, hinit=180))

    dae.p['theta'] = np.array([1])
    soltheta1 = Rodas(dae, [0, 20 * 3600], y0, Opt(fix_h=True, hinit=180))

    res = pd.read_excel(datadir / 'res.xlsx',
                        sheet_name='Tout',
                        engine='openpyxl',
                        index_col=None
                        )
    bench_rodas_theta2 = res['rodastheta2']
    bench_rodas_theta1 = res['rodastheta1']

    rmsetheta2 = np.mean(np.abs(Tmeasured - soltheta2.Y['T'][:, -1]) ** 2) ** (1 / 2)
    rmsetheta1 = np.mean(np.abs(Tmeasured - soltheta1.Y['T'][:, -1]) ** 2) ** (1 / 2)
    assert rmsetheta2 < 0.0949
    assert rmsetheta1 < 0.1129
    np.testing.assert_allclose(bench_rodas_theta2, soltheta2.Y['T'][:, -1])
    np.testing.assert_allclose(bench_rodas_theta1, soltheta1.Y['T'][:, -1])

