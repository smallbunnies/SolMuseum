from Solverz import made_numerical, Rodas, Opt, Eqn, TimeSeriesParam
from SolMuseum.dae.pv import pv
import numpy as np


def test_pv(rtol,
            atol):
    # pv_instance
    pi = pv(ux=1.06592457,
            uy=0.18958303,
            ix=0.60628269,
            iy=0.1078317,
            kop=-0.05,
            koi=-10,
            ws=376.99,
            lf=0.005,
            kip=2,
            kii=9,
            Pnom=12384,
            kp=-0.1,
            ki=-0.01,
            udcref=800,
            cpv=1e-4,
            ldc=0.05,
            cdc=5e-3,
            ISC=19.6,
            IM=18,
            Radiation=1000,
            sref=1000,
            Ttemp=25,
            UOC=864,
            UM=688)

    pvmdl = pi.mdl()
    pvmdl.eqn_uxpv = Eqn('ux_pv', pvmdl.ux_pv - 1.06592457)
    pvmdl.eqn_uypv = Eqn('uy_pv', pvmdl.uy_pv - 0.18958303)

    sdae, y0 = pvmdl.create_instance()
    dae = made_numerical(sdae, y0, sparse=True)

    dae.p['Sfluc_pv'] = TimeSeriesParam('Sfluc_pv',
                                        [1, 0.8, 0.8, 0.8],
                                        [0, 0.00001, 0.00002, 20])

    sol = Rodas(dae,
                [0, 20],
                y0,
                Opt(pbar=True))

    np.testing.assert_allclose(dae.F(sol.T[-1], sol.Y[-1], dae.p),
                               np.array([-4.48678072e-07, -6.92144523e-08, 4.57165470e-08, 2.95659269e-09,
                                         9.70079470e-08, -4.86308451e-03, 9.38277776e-05, 2.97815461e-07,
                                         -5.01186393e-02, 0.00000000e+00, 0.00000000e+00]),
                               atol=rtol,
                               rtol=atol)
