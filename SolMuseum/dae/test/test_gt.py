from Solverz import made_numerical, Rodas, Opt, Eqn
from ..gt import gt

# gt_instance
mi = gt(ux=1.03554323,
        uy=0.07647195,
        ix=0.28211596,
        iy=-0.26874871,
        ra=0,
        xdp=0.0608,
        xqp=0.0969,
        xq=0.0969,
        Damping=10,
        Tj=47.28,
        A=-0.158,
        B=1.158,
        C=0.5,
        D=0.5,
        E=313,
        W=320,
        kp=0.11,
        ki=1/30,
        K1=0.85,
        K2=0.15,
        TRbase=800,
        wref=1,
        qmin=-0.13,
        qmax=1.5,
        T1=12.2,
        T2=1.7,
        TCD=0.16,
        TG=0.05,
        b=0.04,
        TFS=1000,
        )

mimdl = mi.mdl()
mimdl.eqn_ux_syn = Eqn('ux_gt', mimdl.ux_gt - 1.03554323)
mimdl.eqn_uy_syn = Eqn('uy_gt', mimdl.uy_gt - 0.07647195)

sdae, y0 = mimdl.create_instance()
dae = made_numerical(sdae, y0, sparse=True)

sol = Rodas(dae,
            [0, 20],
            y0,
            Opt(pbar=True))

import matplotlib.pyplot as plt

plt.plot(sol.T, sol.Y['omega_gt'])
plt.show()

plt.plot(sol.T, sol.Y['delta_gt'])
plt.show()

plt.plot(sol.T, sol.Y['qfuel_gt'])
plt.show()
