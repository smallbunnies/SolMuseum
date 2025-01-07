from ..synmach import synmach
from Solverz import made_numerical, Rodas, Opt

# syn_mac_instance
mi = synmach(ux=1.0607541902861368,
             uy=0.175893046755461,
             ix=0.503496235830359,
             iy=0.2988361191522479,
             ra=0,
             xdp=0.0608,
             xqp=0.0969,
             xq=0.0969,
             Damping=100,
             Tj=47.28
             )

mimdl = mi.mdl()
mimdl.eqn_ux_syn = Eqn('ux_syn', mimdl.ux_syn - 1.0607541902861368)
mimdl.eqn_uy_syn = Eqn('uy_syn', mimdl.uy_syn - 0.175893046755461)
mimdl.eqn_pm_syn = Eqn('pm_syn', mimdl.Pm_syn - mi.Pm[0])

sdae, y0 = mimdl.create_instance()
dae = made_numerical(sdae, y0, sparse=True)

sol = Rodas(dae,
            [0, 20],
            y0,
            Opt(pbar=True))

import matplotlib.pyplot as plt

plt.plot(sol.T, sol.Y['omega_syn'])
plt.show()

plt.plot(sol.T, sol.Y['delta_syn'])
plt.show()
