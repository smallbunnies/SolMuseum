import numpy as np

from Solverz import Eqn, Opt, Rodas, TimeSeriesParam, made_numerical
from SolMuseum.dae.battery_gfm import battery_gfm


BASE_CASE = dict(
    ux=1.039929869926455,
    uy=-0.20529458257525132,
    ix=0.11178805698717634,
    iy=-0.1643894349994553,
    omega_sys=1.0,
    base_power=25.0,
    system_base_power=100.0,
    active_power=0.6,
    reactive_power=0.16,
    rating=1.1,
    initial_energy=50.0,
    soc_min=5.0,
    soc_max=100.0,
    input_power_min=0.0,
    input_power_max=1.0,
    output_power_min=0.0,
    output_power_max=1.0,
    reactive_power_min=-1.0,
    reactive_power_max=1.0,
    efficiency_in=0.8,
    efficiency_out=0.9,
    omega_ref=1.0,
    omega_lp=500.0,
    kp_pll=0.084,
    ki_pll=4.69,
    Ta=2.0,
    kd=400.0,
    komega=20.0,
    kq=0.2,
    omega_f=1000.0,
    kpv=0.59,
    kiv=736.0,
    kffi=0.0,
    rv=0.0,
    lv=0.2,
    kpc=1.27,
    kic=14.3,
    kffv=0.0,
    omega_ad=50.0,
    kad=0.2,
    dc_voltage=600.0,
    lf=0.08,
    rf=0.003,
    cf=0.074,
    lg=0.2,
    rg=0.01,
    base_frequency=60.0,
)


def _build_model(step=False):
    device = battery_gfm(name='battery', **BASE_CASE)
    mdl = device.mdl()
    if step:
        mdl.u_step_battery = TimeSeriesParam('u_step_battery', [1.0, 1.0, 0.95, 0.95], [0.0, 1.0, 1.0001, 4.0])
        ux_target = BASE_CASE['ux'] * mdl.u_step_battery
        uy_target = BASE_CASE['uy'] * mdl.u_step_battery
    else:
        ux_target = BASE_CASE['ux']
        uy_target = BASE_CASE['uy']
    mdl.eqn_ux_hold = Eqn('eqn_ux_hold_battery', mdl.ux_battery - ux_target)
    mdl.eqn_uy_hold = Eqn('eqn_uy_hold_battery', mdl.uy_battery - uy_target)
    mdl.eqn_omega_hold = Eqn('eqn_omega_hold_battery', mdl.omega_sys_battery - BASE_CASE['omega_sys'])
    return mdl


def test_battery_gfm_equilibrium(rtol, atol):
    mdl = _build_model(step=False)
    sdae, y0 = mdl.create_instance()
    dae = made_numerical(sdae, y0, sparse=True)
    sol = Rodas(dae, [0.0, 0.5], y0, Opt(pbar=False))

    np.testing.assert_allclose(
        dae.F(sol.T[-1], sol.Y[-1], dae.p),
        np.array([ 0.00000000e+00, -1.94650610e-14,  2.67693366e-16,  1.67417688e-13,
                    1.67417688e-13, -1.43773882e-14,  3.33066907e-13,  4.44089210e-16,
                   -4.19803081e-16, -4.44089210e-16, -3.99506817e-15, -1.11022302e-14,
                   -8.39606162e-14, -3.13908165e-12, -1.43874576e-12,  0.00000000e+00,
                    5.65600297e-13,  8.37088440e-13, -1.04636055e-13, -1.38777878e-17,
                    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]),
        rtol=rtol,
        atol=atol,
    )

    np.testing.assert_allclose(
        sol.Y[-1],
        np.array([ 1.03992987e+00, -2.05294583e-01,  1.11788057e-01, -1.64389435e-01,
                    1.00000000e+00,  1.18227016e+00,  3.16485878e-16,  7.87340549e-17,
                   -1.03749253e-01, -2.52421903e-02,  1.00000000e+00,  7.18481464e-01,
                    6.29899749e-04, -8.77802120e-04,  8.25202835e-02, -6.60124297e-03,
                    1.17862865e+00, -9.27212430e-02,  4.56212767e-01, -5.70540182e-01,
                    1.17591294e+00, -1.22439714e-01,  4.47152228e-01, -6.57557740e-01]),
        rtol=rtol,
        atol=atol,
    )


def test_battery_gfm_voltage_step(rtol, atol):
    mdl = _build_model(step=True)
    sdae, y0 = mdl.create_instance()
    dae = made_numerical(sdae, y0, sparse=True)
    sol = Rodas(dae, [0.0, 4.0], y0, Opt(pbar=False))

    np.testing.assert_allclose(
        dae.F(sol.T[-1], sol.Y[-1], dae.p),
        np.array([-2.63506994e-08, -4.11180788e-07, -8.79287142e-09,  7.77873264e-07,
                    8.07471162e-07, -7.03645836e-08,  2.19765428e-07,  2.69647416e-09,
                   -9.88868668e-10, -1.71420078e-09, -1.42111831e-08, -7.78892950e-08,
                   -5.38223049e-07,  2.51951169e-06, -2.26027678e-07,  2.44777669e-08,
                    5.07788028e-07,  2.55724177e-06, -2.24437903e-07,  0.00000000e+00,
                   -2.77555756e-17,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]),
        rtol=rtol,
        atol=atol,
    )

    np.testing.assert_allclose(
        sol.Y[-1],
        np.array([ 9.87933376e-01, -1.95029853e-01,  1.14650758e-01, -1.86611334e-01,
                    1.00000000e+00,  1.14716930e+00, -1.00869122e-08,  5.97435836e-10,
                   -9.67133315e-02, -1.47562148e-02,  1.00000000e+00,  8.01496458e-01,
                    6.37999193e-04, -1.00488735e-03,  8.00523259e-02, -6.70478600e-03,
                    1.14331871e+00, -9.39134704e-02,  4.66800284e-01, -6.61951511e-01,
                    1.14180848e+00, -1.10773701e-01,  4.58603030e-01, -7.46445338e-01]),
        rtol=rtol,
        atol=atol,
    )
