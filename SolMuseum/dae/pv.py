from Solverz import Ode, TimeSeriesParam, Param, Model, Saturation, exp, ln
from Solverz import Var
from Solverz.utilities.type_checker import is_number
from Solverz.num_api.Array import Array
from sympy import re as real, im as imag
import numpy as np
from warnings import warn
from ..util import rename_mdl


class pv:

    def __init__(self,
                 name='pv',
                 **kwargs):

        self.name = name

        for key, value in kwargs.items():
            if isinstance(value, list) or is_number(value):
                kwargs[key] = Array(value, dim=1)

        # PV panel parameter
        self.ISC = kwargs.get('ISC')
        self.IM = kwargs.get('IM')
        self.Radiation = kwargs.get('Radiation')
        # self.Sfluc = Sfluc
        self.sref = kwargs.get('sref')
        self.Ttemp = kwargs.get('Ttemp')
        self.UOC = kwargs.get('UOC')
        self.UM = kwargs.get('UM')

        # PV grid variable
        self.ux = kwargs.get('ux')
        self.uy = kwargs.get('uy')
        self.ix = kwargs.get('ix')
        self.iy = kwargs.get('iy')

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
        isc = self.ISC * self.Radiation / self.sref * (1 + 0.0025 * (self.Ttemp - 25))
        im = self.IM * self.Radiation / self.sref * (1 + 0.0025 * (self.Ttemp - 25))
        uoc = self.UOC * (1 - 0.00288 * (self.Ttemp - 25)) * np.log(np.exp(1) + 0.5 * (self.Radiation / 1000 - 1))
        um = self.UM * (1 - 0.00288 * (self.Ttemp - 25)) * np.log(np.exp(1) + 0.5 * (self.Radiation / 1000 - 1))
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
        P_actual = self.ux * self.ix + self.uy * self.iy

        if not np.all(np.isclose(P, P_actual, atol=1e-6, rtol=1e-4)):
            warn(f"PV output and grid injection not match, with deviation {np.abs(P - P_actual)}. "
                 "Please calculate PV output and perform the power flow first!")

        ugrid_pv = self.ux + 1j * self.uy
        exp_j_theta = ugrid_pv / np.abs(ugrid_pv)
        usk_ctrl = ugrid_pv / exp_j_theta
        ud_pv = usk_ctrl.real
        uq_pv = usk_ctrl.imag

        ig = self.ix + 1j * self.iy
        ig_ctrl = ig / exp_j_theta
        id_pv = ig_ctrl.real
        iq_pv = ig_ctrl.imag

        self.iL = self.ipv
        self.uix = self.ux - self.lf * self.iy
        self.uiy = self.uy + self.lf * self.ix
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

        name = self.name

        m = Model()

        # panel model
        m.Radiation = Param('Radiation_'+name, self.Radiation)
        m.ISC = Param('ISC_'+name, self.ISC)
        m.Sfluc = TimeSeriesParam('Sfluc_'+name, [1, 1], [0, 3600])
        m.sref = Param('sref_'+name, self.sref)
        m.Ttemp = Param('Ttemp_'+name, self.Ttemp)
        m.IM = Param('IM_'+name, self.IM)
        m.UOC = Param('UOC_'+name, self.UOC)
        m.UM = Param('UM_'+name, self.UM)

        isc = m.ISC * m.Radiation * m.Sfluc / m.sref * (1 + 0.0025 * (m.Ttemp - 25))
        im = m.IM * m.Radiation * m.Sfluc / m.sref * (1 + 0.0025 * (m.Ttemp - 25))
        uoc = m.UOC * (1 - 0.00288 * (m.Ttemp - 25)) * ln(exp(1) + 0.5 * (m.Radiation * m.Sfluc / 1000 - 1))
        um = m.UM * (1 - 0.00288 * (m.Ttemp - 25)) * ln(exp(1) + 0.5 * (m.Radiation * m.Sfluc / 1000 - 1))
        c2 = (um / uoc - 1) / ln(1 - im / isc)
        c1 = (1 - im / isc) * exp(-um / (c2 * uoc))

        # circuit model
        m.kop = Param('kop_'+name, self.kop)
        m.koi = Param('koi_'+name, self.koi)
        m.ws = Param('ws_'+name, self.ws)
        m.lf = Param('lf_'+name, self.lf)
        m.kip = Param('kip_'+name, self.kip)
        m.kii = Param('kii_'+name, self.kii)
        m.Pnom = Param('Pnom_'+name, self.Pnom)
        m.kp = Param('kp_'+name, self.kp)
        m.ki = Param('ki_'+name, self.ki)
        m.udcref = Param('udcref_'+name, self.udcref)
        m.cpv = Param('cpv_'+name, self.cpv)
        m.ldc = Param('ldc_'+name, self.ldc)
        m.cdc = Param('cdc_'+name, self.cdc)

        # perform initialization
        m.ux = Var('ux_'+name, self.ux)
        m.uy = Var('uy_'+name, self.uy)
        m.ix = Var('ix_'+name, self.ix)
        m.iy = Var('iy_'+name, self.iy)
        m.udc = Var('udc_'+name, self.udc)
        m.idref1 = Var('idref1_'+name, self.idref1)
        m.urd1 = Var('urd1_'+name, self.urd1)
        m.urq1 = Var('urq1_'+name, self.urq1)
        m.upv = Var('upv_'+name, self.upv)
        m.D1 = Var('D1_'+name, self.D1)
        m.iL = Var('iL_'+name, self.iL)

        ugrid_pv = m.ux + 1j * m.uy
        exp_j_theta = ugrid_pv / abs(ugrid_pv)
        usk_ctrl = ugrid_pv / exp_j_theta
        ud_pv = real(usk_ctrl)
        uq_pv = imag(usk_ctrl)

        ig = m.ix + 1j * m.iy
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

        m.eqn_id = Ode('eqn_id_'+name, m.ws / m.lf * (uix - m.ux + m.lf * m.iy), m.ix)
        m.eqn_iq = Ode('eqn_iq_'+name, m.ws / m.lf * (uiy - m.uy - m.lf * m.ix), m.iy)
        m.eqn_idref1 = Ode('eqn_idref1_'+name, m.udcref - m.udc, m.idref1)
        m.eqn_urd1 = Ode('eqn_urd1_'+name,
                         idref - id_pv,
                         m.urd1)
        m.eqn_urq1 = Ode('eqn_urq1_'+name,
                         iqref - iq_pv,
                         m.urq1)
        m.eqn_upv = Ode('eqn_upv_'+name,
                        1 / m.cpv * (ipv - m.iL),
                        m.upv)
        m.eqn_iL = Ode('eqn_iL_'+name,
                       1 / m.ldc * (m.upv - (1 - D) * m.udc),
                       m.iL)
        m.eqn_udc = Ode('eqn_udc_'+name,
                        1 / m.cdc * ((1 - D) * m.iL - idc),
                        m.udc)
        m.eqn_D1 = Ode('eqn_D1_'+name,
                       um - m.upv,
                       m.D1)

        m = rename_mdl(m, name)

        return m
