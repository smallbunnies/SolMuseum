from Solverz import made_numerical, Rodas, Opt, Eqn, TimeSeriesParam
from ..pv import pv

# pv_instance
pi = pv(ux_pv=1.06592457,
        uy_pv=0.18958303,
        ix_pv=0.60628269,
        iy_pv=0.1078317,
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
        cdc=5e-3)

pvmdl = pi.mdl()
pvmdl.eqn_uxpv = Eqn('uxpv', pvmdl.ux_pv - 1.06592457)
pvmdl.eqn_uypv = Eqn('uypv', pvmdl.uy_pv - 0.18958303)

sdae, y0 = pvmdl.create_instance()
dae = made_numerical(sdae, y0, sparse=True)

dae.p['Sfluc'] = TimeSeriesParam('Sfluc',
                                 [1, 0.8, 0.8, 0.8],
                                 [0, 0.00001, 0.00002, 20])

sol = Rodas(dae,
            [0, 20],
            y0,
            Opt(pbar=True))

import matplotlib.pyplot as plt
plt.plot(sol.T, sol.Y['iL'])
plt.show()

plt.plot(sol.T, sol.Y['D1'])
plt.show()
