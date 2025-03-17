from SolMuseum.dae.synmach import synmach
from Solverz import made_numerical, Rodas, Opt, Eqn


def test_syn():
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
                 Tj=47.28,
                 use_coi=False
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

    import numpy as np
    assert np.max(np.abs(dae.F(sol.T[-1], sol.Y[-1], dae.p))) < 1e-15
