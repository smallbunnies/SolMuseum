from Solverz import made_numerical, Rodas, Opt, Eqn, TimeSeriesParam
from SolMuseum.dae.st import st
import numpy as np

z = 1e-7
eta = 0.9
F = 1.1
phi = (eta * F - 0.27159153) / z


def test_st():
    # st_instance
    mi = st(ux=1.03554323,
            uy=0.07647195,
            ix=0.28211596,
            iy=-0.26874871,
            ra=0,
            xdp=0.0608,
            xqp=0.0969,
            xq=0.0969,
            Damping=10,
            Tj=47.28,
            phi=phi,
            z=z,
            F=F,
            eta=eta,
            TREF=85,
            alpha=0.3,
            mu_min=0,
            mu_max=1,
            TCH=0.2,
            TRH=5,
            kp=-1,
            ki=-1,
            use_coi=False
            )

    mimdl = mi.mdl()
    mimdl.eqn_ux_syn = Eqn('ux_st', mimdl.ux_st - 1.03554323)
    mimdl.eqn_uy_syn = Eqn('uy_st', mimdl.uy_st - 0.07647195)
    mimdl.Ts_real = TimeSeriesParam('Ts_real',
                                    [85, 90, 90, 90, 85, 85],
                                    [0, 1, 2, 10, 11, 20])
    mimdl.eqn_Ts = Eqn('Ts', mimdl.Ts_st - mimdl.Ts_real)

    sdae, y0 = mimdl.create_instance()
    dae, code = made_numerical(sdae, y0, sparse=True, output_code=True)

    sol = Rodas(dae,
                [0, 100],
                y0,
                Opt(pbar=True))

    np.testing.assert_allclose(sol.Y[18],
                               np.array([1.03554323e+00, 7.64719500e-02, 6.60655484e-01, -2.34189651e-01,
                                         1.31986358e-01, 1.00009051e+00, 5.79501837e-01, 9.97910538e-01,
                                         2.80128676e-01, 8.99405943e+00, 4.10498163e+06, -3.99405943e+00,
                                         9.00000000e+01]),
                               atol=1e-6,
                               rtol=1e-6)
