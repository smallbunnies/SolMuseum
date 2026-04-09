import warnings

import numpy as np
from scipy.optimize import root

from Solverz import Eqn, Model, Ode, Param, Var, atan2, cos, sin
from SolMuseum.util import rename_mdl


INIT_TOL = 1e-10


def _ri_dq_np(theta, vr, vi):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([s * vr - c * vi, c * vr + s * vi], dtype=float)


def _dq_ri_np(theta, vd, vq):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([s * vd + c * vq, -c * vd + s * vq], dtype=float)


def _ri_dq_sym(theta, vr, vi):
    return sin(theta) * vr - cos(theta) * vi, cos(theta) * vr + sin(theta) * vi


def _dq_ri_sym(theta, vd, vq):
    return sin(theta) * vd + cos(theta) * vq, -cos(theta) * vd + sin(theta) * vq


class battery_gfm:

    def __init__(self, name='battery_gfm', **kwargs):
        self.name = name

        self.ux = kwargs.get('ux')
        self.uy = kwargs.get('uy')
        self.ix = kwargs.get('ix')
        self.iy = kwargs.get('iy')
        self.omega_sys = kwargs.get('omega_sys', 1.0)

        self.base_power = kwargs.get('base_power')
        self.system_base_power = kwargs.get('system_base_power', 100.0)
        self.active_power = kwargs.get('active_power')
        self.reactive_power = kwargs.get('reactive_power')
        self.rating = kwargs.get('rating')
        self.initial_energy = kwargs.get('initial_energy')
        self.soc_min = kwargs.get('soc_min')
        self.soc_max = kwargs.get('soc_max')
        self.input_power_min = kwargs.get('input_power_min')
        self.input_power_max = kwargs.get('input_power_max')
        self.output_power_min = kwargs.get('output_power_min')
        self.output_power_max = kwargs.get('output_power_max')
        self.reactive_power_min = kwargs.get('reactive_power_min')
        self.reactive_power_max = kwargs.get('reactive_power_max')
        self.efficiency_in = kwargs.get('efficiency_in')
        self.efficiency_out = kwargs.get('efficiency_out')

        self.p_ref = kwargs.get('p_ref')
        self.q_ref = kwargs.get('q_ref')
        self.v_ref = kwargs.get('v_ref')
        self.omega_ref = kwargs.get('omega_ref', 1.0)

        self.omega_lp = kwargs.get('omega_lp')
        self.kp_pll = kwargs.get('kp_pll')
        self.ki_pll = kwargs.get('ki_pll')
        self.Ta = kwargs.get('Ta')
        self.kd = kwargs.get('kd')
        self.komega = kwargs.get('komega')
        self.kq = kwargs.get('kq')
        self.omega_f = kwargs.get('omega_f')
        self.kpv = kwargs.get('kpv')
        self.kiv = kwargs.get('kiv')
        self.kffi = kwargs.get('kffi')
        self.rv = kwargs.get('rv')
        self.lv = kwargs.get('lv')
        self.kpc = kwargs.get('kpc')
        self.kic = kwargs.get('kic')
        self.kffv = kwargs.get('kffv', 0.0)
        self.omega_ad = kwargs.get('omega_ad')
        self.kad = kwargs.get('kad')
        self.dc_voltage = kwargs.get('dc_voltage')
        self.lf = kwargs.get('lf')
        self.rf = kwargs.get('rf')
        self.cf = kwargs.get('cf')
        self.lg = kwargs.get('lg')
        self.rg = kwargs.get('rg')
        self.base_frequency = kwargs.get('base_frequency', 60.0)

        self._check_required()
        self.current_scale = self.base_power / self.system_base_power
        self.omega_b = 2.0 * np.pi * self.base_frequency
        self._initialized = False

    def _check_required(self):
        required = [
            'ux', 'uy', 'ix', 'iy', 'base_power', 'omega_lp', 'kp_pll', 'ki_pll', 'Ta', 'kd', 'komega', 'kq',
            'omega_f', 'kpv', 'kiv', 'kffi', 'rv', 'lv', 'kpc', 'kic', 'omega_ad', 'kad', 'dc_voltage', 'lf',
            'rf', 'cf', 'lg', 'rg'
        ]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(f'Missing battery_gfm parameters: {", ".join(missing)}')

    def _solve_init(self, residual, x0, label):
        sol = root(residual, x0, method='hybr', options={'xtol': INIT_TOL, 'maxfev': 4000})
        if not sol.success:
            warnings.warn(f'battery_gfm {label} initialization did not fully converge: {sol.message}')
        return sol.x

    def _initialize_filter(self):
        ir_filter = self.ix / self.current_scale
        ii_filter = self.iy / self.current_scale

        def residual(x):
            vr_cnv, vi_cnv, ir_cnv, ii_cnv, vr_filter, vi_filter = x
            return np.array([
                vr_cnv - vr_filter - self.rf * ir_cnv + self.omega_sys * self.lf * ii_cnv,
                vi_cnv - vi_filter - self.rf * ii_cnv - self.omega_sys * self.lf * ir_cnv,
                ir_cnv - ir_filter + self.omega_sys * self.cf * vi_filter,
                ii_cnv - ii_filter - self.omega_sys * self.cf * vr_filter,
                vr_filter - self.ux - self.rg * ir_filter + self.omega_sys * self.lg * ii_filter,
                vi_filter - self.uy - self.rg * ii_filter - self.omega_sys * self.lg * ir_filter,
            ], dtype=float)

        x0 = np.array([self.ux, self.uy, ir_filter, ii_filter, self.ux, self.uy], dtype=float)
        sol = self._solve_init(residual, x0, 'filter')
        self.Vr_cnv0, self.Vi_cnv0, self.Ir_cnv0, self.Ii_cnv0, self.Vr_filter0, self.Vi_filter0 = sol
        self.Ir_filter0 = ir_filter
        self.Ii_filter0 = ii_filter

    def _initialize_pll(self):
        def residual(x):
            vpll_d, vpll_q, eps_pll, theta_pll = x
            vd_pll, vq_pll = _ri_dq_np(theta_pll + np.pi / 2.0, self.Vr_filter0, self.Vi_filter0)
            angle = np.atan2(vpll_q, vpll_d)
            return np.array([
                vd_pll - vpll_d,
                vq_pll - vpll_q,
                angle,
                self.kp_pll * angle + self.ki_pll * eps_pll,
            ], dtype=float)

        x0 = np.array([
            self.Vr_filter0,
            0.0,
            0.0,
            np.atan2(self.Vi_filter0, self.Vr_filter0),
        ], dtype=float)
        sol = self._solve_init(residual, x0, 'pll')
        self.vpll_d0, self.vpll_q0, self.eps_pll0, self.theta_pll0 = sol
        pll_angle = np.atan2(self.vpll_q0, self.vpll_d0)
        self.omega_pll0 = 1.0 + self.kp_pll * pll_angle + self.ki_pll * self.eps_pll0

    def _initialize_outer(self):
        self.p_elec0 = self.Ir_filter0 * self.Vr_filter0 + self.Ii_filter0 * self.Vi_filter0
        self.q_elec0 = -self.Ii_filter0 * self.Vr_filter0 + self.Ir_filter0 * self.Vi_filter0
        self.theta_oc0 = np.atan2(self.Vi_cnv0, self.Vr_cnv0)
        self.omega_oc0 = self.omega_ref
        self.q_m0 = self.q_elec0
        self.p_ref0 = self.p_elec0 if self.p_ref is None else self.p_ref
        self.q_ref0 = self.q_elec0 if self.q_ref is None else self.q_ref
        self.v_ref_guess = 1.0 if self.v_ref is None else self.v_ref
        self.V_oc0 = self.v_ref_guess + self.kq * (self.q_ref0 - self.q_m0)

    def _initialize_inner(self):
        def residual(x):
            theta_oc, v_refr, xi_d, xi_q, gamma_d, gamma_q, phi_d, phi_q = x
            id_filter, iq_filter = _ri_dq_np(theta_oc + np.pi / 2.0, self.Ir_filter0, self.Ii_filter0)
            id_cnv, iq_cnv = _ri_dq_np(theta_oc + np.pi / 2.0, self.Ir_cnv0, self.Ii_cnv0)
            vd_filter, vq_filter = _ri_dq_np(theta_oc + np.pi / 2.0, self.Vr_filter0, self.Vi_filter0)
            vd_cnv0, vq_cnv0 = _ri_dq_np(theta_oc + np.pi / 2.0, self.Vr_cnv0, self.Vi_cnv0)

            vd_filter_ref = v_refr - self.rv * id_filter + self.omega_oc0 * self.lv * iq_filter
            vq_filter_ref = -self.rv * iq_filter - self.omega_oc0 * self.lv * id_filter

            id_pi = self.kpv * (vd_filter_ref - vd_filter) + self.kiv * xi_d
            iq_pi = self.kpv * (vq_filter_ref - vq_filter) + self.kiv * xi_q

            id_cnv_ref = id_pi - self.cf * self.omega_oc0 * vq_filter + self.kffi * id_filter
            iq_cnv_ref = iq_pi + self.cf * self.omega_oc0 * vd_filter + self.kffi * iq_filter

            vd_pi = self.kpc * (id_cnv_ref - id_cnv) + self.kic * gamma_d
            vq_pi = self.kpc * (iq_cnv_ref - iq_cnv) + self.kic * gamma_q

            vd_cnv_ref = (
                vd_pi
                - self.omega_oc0 * self.lf * iq_cnv
                + self.kffv * vd_filter
                - self.kad * (vd_filter - phi_d)
            )
            vq_cnv_ref = (
                vq_pi
                + self.omega_oc0 * self.lf * id_cnv
                + self.kffv * vq_filter
                - self.kad * (vq_filter - phi_q)
            )

            return np.array([
                vd_filter_ref - vd_filter,
                vq_filter_ref - vq_filter,
                id_cnv_ref - id_cnv,
                iq_cnv_ref - iq_cnv,
                vd_filter - phi_d,
                vq_filter - phi_q,
                vd_cnv_ref - vd_cnv0,
                vq_cnv_ref - vq_cnv0,
            ], dtype=float)

        x0 = np.array([self.theta_oc0, self.v_ref_guess, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        sol = self._solve_init(residual, x0, 'inner')
        (
            self.theta_oc0,
            self.v_ref0,
            self.xi_d0,
            self.xi_q0,
            self.gamma_d0,
            self.gamma_q0,
            self.phi_d0,
            self.phi_q0,
        ) = sol
        self.V_oc0 = self.v_ref0 + self.kq * (self.q_ref0 - self.q_m0)
        self.md0, self.mq0 = _ri_dq_np(self.theta_oc0 + np.pi / 2.0, self.Vr_cnv0, self.Vi_cnv0) / self.dc_voltage

    def _initialize(self):
        if self._initialized:
            return
        self._initialize_filter()
        self._initialize_pll()
        self._initialize_outer()
        self.Vdc0 = self.dc_voltage
        self._initialize_inner()
        self._initialized = True

    def mdl(self):
        self._initialize()
        name = self.name
        m = Model()

        m.base_power = Param('base_power_' + name, self.base_power)
        m.system_base_power = Param('system_base_power_' + name, self.system_base_power)
        m.omega_b = Param('omega_b_' + name, self.omega_b)
        m.omega_lp = Param('omega_lp_' + name, self.omega_lp)
        m.kp_pll = Param('kp_pll_' + name, self.kp_pll)
        m.ki_pll = Param('ki_pll_' + name, self.ki_pll)
        m.Ta = Param('Ta_' + name, self.Ta)
        m.kd = Param('kd_' + name, self.kd)
        m.komega = Param('komega_' + name, self.komega)
        m.kq = Param('kq_' + name, self.kq)
        m.omega_f = Param('omega_f_' + name, self.omega_f)
        m.kpv = Param('kpv_' + name, self.kpv)
        m.kiv = Param('kiv_' + name, self.kiv)
        m.kffi = Param('kffi_' + name, self.kffi)
        m.rv = Param('rv_' + name, self.rv)
        m.lv = Param('lv_' + name, self.lv)
        m.kpc = Param('kpc_' + name, self.kpc)
        m.kic = Param('kic_' + name, self.kic)
        m.kffv = Param('kffv_' + name, self.kffv)
        m.omega_ad = Param('omega_ad_' + name, self.omega_ad)
        m.kad = Param('kad_' + name, self.kad)
        m.dc_voltage = Param('dc_voltage_' + name, self.dc_voltage)
        m.lf = Param('lf_' + name, self.lf)
        m.rf = Param('rf_' + name, self.rf)
        m.cf = Param('cf_' + name, self.cf)
        m.lg = Param('lg_' + name, self.lg)
        m.rg = Param('rg_' + name, self.rg)
        m.p_ref = Param('p_ref_' + name, self.p_ref0)
        m.q_ref = Param('q_ref_' + name, self.q_ref0)
        m.v_ref = Param('v_ref_' + name, self.v_ref0)
        m.omega_ref = Param('omega_ref_' + name, self.omega_ref)

        m.ux = Var('ux_' + name, self.ux)
        m.uy = Var('uy_' + name, self.uy)
        m.ix = Var('ix_' + name, self.ix)
        m.iy = Var('iy_' + name, self.iy)
        m.omega_sys = Var('omega_sys_' + name, self.omega_sys)

        m.vpll_d = Var('vpll_d_' + name, self.vpll_d0)
        m.vpll_q = Var('vpll_q_' + name, self.vpll_q0)
        m.eps_pll = Var('eps_pll_' + name, self.eps_pll0)
        m.theta_pll = Var('theta_pll_' + name, self.theta_pll0)

        m.theta_oc = Var('theta_oc_' + name, self.theta_oc0)
        m.omega_oc = Var('omega_oc_' + name, self.omega_oc0)
        m.q_m = Var('q_m_' + name, self.q_m0)

        m.xi_d = Var('xi_d_' + name, self.xi_d0)
        m.xi_q = Var('xi_q_' + name, self.xi_q0)
        m.gamma_d = Var('gamma_d_' + name, self.gamma_d0)
        m.gamma_q = Var('gamma_q_' + name, self.gamma_q0)
        m.phi_d = Var('phi_d_' + name, self.phi_d0)
        m.phi_q = Var('phi_q_' + name, self.phi_q0)

        m.Ir_cnv = Var('Ir_cnv_' + name, self.Ir_cnv0)
        m.Ii_cnv = Var('Ii_cnv_' + name, self.Ii_cnv0)
        m.Vr_filter = Var('Vr_filter_' + name, self.Vr_filter0)
        m.Vi_filter = Var('Vi_filter_' + name, self.Vi_filter0)
        m.Ir_filter = Var('Ir_filter_' + name, self.Ir_filter0)
        m.Ii_filter = Var('Ii_filter_' + name, self.Ii_filter0)

        pll_theta = m.theta_pll + np.pi / 2.0
        oc_theta = m.theta_oc + np.pi / 2.0

        vd_pll, vq_pll = _ri_dq_sym(pll_theta, m.Vr_filter, m.Vi_filter)
        pll_angle = atan2(m.vpll_q, m.vpll_d)
        omega_pll = 1.0 + m.kp_pll * pll_angle + m.ki_pll * m.eps_pll

        p_elec = m.Ir_filter * m.Vr_filter + m.Ii_filter * m.Vi_filter
        q_elec = -m.Ii_filter * m.Vr_filter + m.Ir_filter * m.Vi_filter
        v_oc = m.v_ref + m.kq * (m.q_ref - m.q_m)

        id_filter, iq_filter = _ri_dq_sym(oc_theta, m.Ir_filter, m.Ii_filter)
        id_cnv, iq_cnv = _ri_dq_sym(oc_theta, m.Ir_cnv, m.Ii_cnv)
        vd_filter, vq_filter = _ri_dq_sym(oc_theta, m.Vr_filter, m.Vi_filter)

        vd_filter_ref = v_oc - m.rv * id_filter + m.omega_oc * m.lv * iq_filter
        vq_filter_ref = -m.rv * iq_filter - m.omega_oc * m.lv * id_filter

        id_pi = m.kpv * (vd_filter_ref - vd_filter) + m.kiv * m.xi_d
        iq_pi = m.kpv * (vq_filter_ref - vq_filter) + m.kiv * m.xi_q

        id_cnv_ref = id_pi - m.cf * m.omega_oc * vq_filter + m.kffi * id_filter
        iq_cnv_ref = iq_pi + m.cf * m.omega_oc * vd_filter + m.kffi * iq_filter

        vd_pi = m.kpc * (id_cnv_ref - id_cnv) + m.kic * m.gamma_d
        vq_pi = m.kpc * (iq_cnv_ref - iq_cnv) + m.kic * m.gamma_q

        vd_cnv_ref = vd_pi - m.omega_oc * m.lf * iq_cnv + m.kffv * vd_filter - m.kad * (vd_filter - m.phi_d)
        vq_cnv_ref = vq_pi + m.omega_oc * m.lf * id_cnv + m.kffv * vq_filter - m.kad * (vq_filter - m.phi_q)

        md = vd_cnv_ref / m.dc_voltage
        mq = vq_cnv_ref / m.dc_voltage
        vr_cnv, vi_cnv = _dq_ri_sym(oc_theta, md, mq)
        vr_cnv = vr_cnv * m.dc_voltage
        vi_cnv = vi_cnv * m.dc_voltage

        m.eqn_vpll_d = Ode('eqn_vpll_d_' + name, m.omega_lp * (vd_pll - m.vpll_d), m.vpll_d)
        m.eqn_vpll_q = Ode('eqn_vpll_q_' + name, m.omega_lp * (vq_pll - m.vpll_q), m.vpll_q)
        m.eqn_eps_pll = Ode('eqn_eps_pll_' + name, pll_angle, m.eps_pll)
        m.eqn_theta_pll = Ode('eqn_theta_pll_' + name, m.omega_b * (omega_pll - m.omega_sys), m.theta_pll)

        m.eqn_theta_oc = Ode('eqn_theta_oc_' + name, m.omega_b * (m.omega_oc - m.omega_sys), m.theta_oc)
        m.eqn_omega_oc = Ode(
            'eqn_omega_oc_' + name,
            (m.p_ref - p_elec - m.kd * (m.omega_oc - omega_pll) - m.komega * (m.omega_oc - m.omega_ref)) / m.Ta,
            m.omega_oc,
        )
        m.eqn_q_m = Ode('eqn_q_m_' + name, m.omega_f * (q_elec - m.q_m), m.q_m)

        m.eqn_xi_d = Ode('eqn_xi_d_' + name, vd_filter_ref - vd_filter, m.xi_d)
        m.eqn_xi_q = Ode('eqn_xi_q_' + name, vq_filter_ref - vq_filter, m.xi_q)
        m.eqn_gamma_d = Ode('eqn_gamma_d_' + name, id_cnv_ref - id_cnv, m.gamma_d)
        m.eqn_gamma_q = Ode('eqn_gamma_q_' + name, iq_cnv_ref - iq_cnv, m.gamma_q)
        m.eqn_phi_d = Ode('eqn_phi_d_' + name, m.omega_ad * (vd_filter - m.phi_d), m.phi_d)
        m.eqn_phi_q = Ode('eqn_phi_q_' + name, m.omega_ad * (vq_filter - m.phi_q), m.phi_q)

        m.eqn_Ir_cnv = Ode(
            'eqn_Ir_cnv_' + name,
            m.omega_b / m.lf * (vr_cnv - m.Vr_filter - m.rf * m.Ir_cnv + m.lf * m.omega_sys * m.Ii_cnv),
            m.Ir_cnv,
        )
        m.eqn_Ii_cnv = Ode(
            'eqn_Ii_cnv_' + name,
            m.omega_b / m.lf * (vi_cnv - m.Vi_filter - m.rf * m.Ii_cnv - m.lf * m.omega_sys * m.Ir_cnv),
            m.Ii_cnv,
        )
        m.eqn_Vr_filter = Ode(
            'eqn_Vr_filter_' + name,
            m.omega_b / m.cf * (m.Ir_cnv - m.Ir_filter + m.cf * m.omega_sys * m.Vi_filter),
            m.Vr_filter,
        )
        m.eqn_Vi_filter = Ode(
            'eqn_Vi_filter_' + name,
            m.omega_b / m.cf * (m.Ii_cnv - m.Ii_filter - m.cf * m.omega_sys * m.Vr_filter),
            m.Vi_filter,
        )
        m.eqn_Ir_filter = Ode(
            'eqn_Ir_filter_' + name,
            m.omega_b / m.lg * (m.Vr_filter - m.ux - m.rg * m.Ir_filter + m.lg * m.omega_sys * m.Ii_filter),
            m.Ir_filter,
        )
        m.eqn_Ii_filter = Ode(
            'eqn_Ii_filter_' + name,
            m.omega_b / m.lg * (m.Vi_filter - m.uy - m.rg * m.Ii_filter - m.lg * m.omega_sys * m.Ir_filter),
            m.Ii_filter,
        )

        m.eqn_ix = Eqn('eqn_ix_' + name, m.ix - m.base_power / m.system_base_power * m.Ir_filter)
        m.eqn_iy = Eqn('eqn_iy_' + name, m.iy - m.base_power / m.system_base_power * m.Ii_filter)

        return rename_mdl(m, name)
