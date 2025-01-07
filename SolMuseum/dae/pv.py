from Solverz import Eqn, Ode, AliasVar, TimeSeriesParam, Param, Model, Saturation, exp, ln
from Solverz import Var, Abs
from Solverz.sym_algebra.symbols import iVar, idx
from Solverz.utilities.type_checker import is_integer, is_number
from sympy import re as real, im as imag
import numpy as np
from warnings import warn


class pv:

    def __init__(self,
                 ISC=19.6,
                 IM=18,
                 Radiation=1000,
                 sref=1000,
                 Ttemp_pv=25,
                 UOC=864,
                 UM=688,
                 **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = np.array(value)
            elif is_number(value):
                kwargs[key] = np.array(value).reshape((-1,))

        # PV panel parameter
        self.ISC = ISC
        self.IM = IM
        self.Radiation = Radiation
        # self.Sfluc = Sfluc
        self.sref = sref
        self.Ttemp_pv = Ttemp_pv
        self.UOC = UOC
        self.UM = UM

        # PV grid variable
        self.ux_pv = kwargs.get('ux_pv')
        self.uy_pv = kwargs.get('uy_pv')
        self.ix_pv = kwargs.get('ix_pv')
        self.iy_pv = kwargs.get('iy_pv')

        # circuit parameter
        self.kop = kwargs.get('kop')
        self.koi = kwargs.get('koi')
        self.ws = kwargs.get('ws')
        self.lf = kwargs.get('lf')
        self.kip = kwargs.get('kip')
        self.kii = kwargs.get('kii')
        self.Pnom = kwargs.get('Pnom')
        self.kp = kwargs.get('kp')
        self.ki = kwargs.get('ki')
        self.udcref = kwargs.get('udcref')
        self.cpv = kwargs.get('cpv')
        self.ldc = kwargs.get('ldc')
        self.cdc = kwargs.get('cdc')

        self.iL = None
        self.udc = None
        self.uix = None
        self.uiy = None
        self.uid = None
        self.uiq = None
        self.idc = None
        self.urd = None
        self.urq = None
        self.upv = None
        self.ipv = None
        self.urd1 = None
        self.urq1 = None
        self.idref1 = None
        self.iqref1 = None
        self.D1 = None

    def cal_P(self):
        # 先计算upv，ipv，
        isc = self.ISC * self.Radiation / self.sref * (1 + 0.0025 * (self.Ttemp_pv - 25))
        im = self.IM * self.Radiation / self.sref * (1 + 0.0025 * (self.Ttemp_pv - 25))
        uoc = self.UOC * (1 - 0.00288 * (self.Ttemp_pv - 25)) * np.log(np.exp(1) + 0.5 * (self.Radiation / 1000 - 1))
        um = self.UM * (1 - 0.00288 * (self.Ttemp_pv - 25)) * np.log(np.exp(1) + 0.5 * (self.Radiation / 1000 - 1))
        if 1 - im / isc != 0:
            c2 = (um / uoc - 1) / np.log(1 - im / isc)
        else:
            c2 = 1
        c1 = (1 - im / isc) * np.exp(-um / (c2 * uoc))
        self.upv = um
        self.ipv = isc * (1 - c1 * (np.exp(self.upv / (c2 * uoc)) - 1))

        return self.upv * self.ipv / (3 / 2) / self.Pnom

    def pv_init(self):
        P = self.cal_P()
        P_actual = self.ux_pv * self.ix_pv + self.uy_pv * self.iy_pv

        if not np.all(np.isclose(P, P_actual, atol=1e-6, rtol=1e-4)):
            warn(f"PV output and grid injection not match, with deviation {np.abs(P-P_actual)}. "
                 "Please calculate PV output and perform the power flow first!")

        ugrid_pv = self.ux_pv + 1j * self.uy_pv
        exp_j_theta = ugrid_pv / np.abs(ugrid_pv)
        usk_ctrl = ugrid_pv / exp_j_theta
        ud_pv = usk_ctrl.real
        uq_pv = usk_ctrl.imag

        ig = self.ix_pv + 1j * self.iy_pv
        ig_ctrl = ig / exp_j_theta
        id_pv = ig_ctrl.real
        iq_pv = ig_ctrl.imag

        self.iL = self.ipv
        self.uix = self.ux_pv - self.lf * self.iy_pv
        self.uiy = self.uy_pv + self.lf * self.ix_pv
        uidq = (self.uix + 1j * self.uiy) / exp_j_theta
        self.uid = uidq.real
        self.uiq = uidq.imag
        self.udc = self.udcref
        self.idc = 3 / 2 * self.Pnom * (self.uid * id_pv + self.uiq * iq_pv) / self.udc

        K_temp = 380 * 2 * np.sqrt(2 / 3)
        temp = np.clip(K_temp / self.udc, 0, 1)
        urdq = uidq * K_temp / temp / self.udc
        self.urd = urdq.real
        self.urq = urdq.imag
        idref = id_pv
        iqref = iq_pv
        self.urd1 = (self.urd - ud_pv + self.ws * self.lf * iq_pv) / self.kii
        self.urq1 = (self.urq - uq_pv - self.ws * self.lf * id_pv) / self.kii
        self.idref1 = idref / self.koi
        self.iqref1 = iqref / self.koi
        D = 1 - self.upv / self.udc
        self.D1 = D / self.ki

    def mdl(self):
        self.pv_init()

        m = Model()

        # panel model
        m.Radiation = Param('Radiation', self.Radiation)
        m.ISC = Param("ISC", self.ISC)
        m.Sfluc = TimeSeriesParam('Sfluc', [1, 1], [0, 3600])
        m.sref = Param("sref", self.sref)
        m.Ttemp_pv = Param("Ttemp_pv", self.Ttemp_pv)
        m.IM = Param("IM", self.IM)
        m.UOC = Param("UOC", self.UOC)
        m.UM = Param("UM", self.UM)

        isc = m.ISC * m.Radiation * m.Sfluc / m.sref * (1 + 0.0025 * (m.Ttemp_pv - 25))
        im = m.IM * m.Radiation * m.Sfluc / m.sref * (1 + 0.0025 * (m.Ttemp_pv - 25))
        uoc = m.UOC * (1 - 0.00288 * (m.Ttemp_pv - 25)) * ln(exp(1) + 0.5 * (m.Radiation * m.Sfluc / 1000 - 1))
        um = m.UM * (1 - 0.00288 * (m.Ttemp_pv - 25)) * ln(exp(1) + 0.5 * (m.Radiation * m.Sfluc / 1000 - 1))
        c2 = (um / uoc - 1) / ln(1 - im / isc)
        c1 = (1 - im / isc) * exp(-um / (c2 * uoc))

        # circuit model
        m.kop = Param('kop', self.kop)
        m.koi = Param('koi', self.koi)
        m.ws = Param('ws', self.ws)
        m.lf = Param('lf', self.lf)
        m.kip = Param('kip', self.kip)
        m.kii = Param('kii', self.kii)
        m.Pnom = Param('Pnom', self.Pnom)
        m.kp = Param('kp', self.kp)
        m.ki = Param('ki', self.ki)
        m.udcref = Param('udcref', self.udcref)
        m.cpv = Param('cpv', self.cpv)
        m.ldc = Param('ldc', self.ldc)
        m.cdc = Param('cdc', self.cdc)

        # perform initialization
        m.ux_pv = Var('ux_pv', self.ux_pv)
        m.uy_pv = Var('uy_pv', self.uy_pv)
        m.ix_pv = Var('ix_pv', self.ix_pv)
        m.iy_pv = Var('iy_pv', self.iy_pv)
        m.udc = Var('udc', self.udc)
        m.idref1 = Var('idref1', self.idref1)
        m.urd1 = Var('urd1', self.urd1)
        m.urq1 = Var('urq1', self.urq1)
        m.upv = Var('upv', self.upv)
        m.D1 = Var('D1', self.D1)
        m.iL = Var('iL', self.iL)

        ugrid_pv = m.ux_pv + 1j * m.uy_pv
        exp_j_theta = ugrid_pv / abs(ugrid_pv)
        usk_ctrl = ugrid_pv / exp_j_theta
        ud_pv = real(usk_ctrl)
        uq_pv = imag(usk_ctrl)

        ig = m.ix_pv + 1j * m.iy_pv
        ig_ctrl = ig / exp_j_theta
        id_pv = real(ig_ctrl)
        iq_pv = imag(ig_ctrl)

        idref = Saturation(m.kop * (m.udcref - m.udc) + m.koi * m.idref1, -1, 1)
        iqref = 0

        urd = ud_pv - m.ws * m.lf * iq_pv + m.kip * (idref - id_pv) + m.kii * m.urd1
        urq = uq_pv + m.ws * m.lf * id_pv + m.kip * (iqref - iq_pv) + m.kii * m.urq1
        urdq = urd + 1j * urq
        K_temp = 380 * 2 * np.sqrt(2 / 3)
        temp = Saturation(K_temp / m.udc, 0, 1)
        uidq = (1 / K_temp) * temp * m.udc * urdq
        uid = real(uidq)
        uiq = imag(uidq)
        uixy = uidq * exp_j_theta
        uix = real(uixy)
        uiy = imag(uixy)
        idc = (3 / 2) * m.Pnom * (uid * id_pv + uiq * iq_pv) / m.udc
        ipv = isc * (1 - c1 * (exp(m.upv / (c2 * uoc)) - 1))
        D = Saturation(m.kp * (um - m.upv) + m.ki * m.D1, 1e-6, 1 - 1e-6)

        m.eqn_id = Ode('eqn_id', m.ws / m.lf * (uix - m.ux_pv + m.lf * m.iy_pv), m.ix_pv)
        m.eqn_iq = Ode('eqn_iq', m.ws / m.lf * (uiy - m.uy_pv - m.lf * m.ix_pv), m.iy_pv)
        m.eqn_idref1 = Ode('eqn_idref1', m.udcref - m.udc, m.idref1)
        m.eqn_urd1 = Ode('eqn_urd1',
                         idref - id_pv,
                         m.urd1)
        m.eqn_urq1 = Ode('eqn_urq1',
                         iqref - iq_pv,
                         m.urq1)
        m.eqn_upv = Ode('eqn_upv',
                        1 / m.cpv * (ipv - m.iL),
                        m.upv)
        m.eqn_iL = Ode('eqn_iL',
                       1 / m.ldc * (m.upv - (1 - D) * m.udc),
                       m.iL)
        m.eqn_udc = Ode('eqn_udc',
                        1 / m.cdc * ((1 - D) * m.iL - idc),
                        m.udc)
        m.eqn_D1 = Ode('eqn_D1',
                       um - m.upv,
                       m.D1)

        # artifact = dict()
        # for key, value in m.__dict__:
        #     if isinstance(value, (Var, Param, TimeSeriesParam, Eqn, Ode)):
        #         artifact[key] = value
        #
        # return artifact
        return m

