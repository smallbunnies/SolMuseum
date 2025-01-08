from Solverz import Eqn, Ode, AliasVar, TimeSeriesParam, Param, Model, Saturation, AntiWindUp, Min
from Solverz import Var, Abs
from Solverz.sym_algebra.symbols import iVar, idx
from Solverz.utilities.type_checker import is_integer, is_number
from sympy import re as real, im as imag
import numpy as np
from warnings import warn

from .synmach import synmach
from .util import rename_mdl


class gt:

    def __init__(self,
                 name='gt',
                 use_coi=True,
                 **kwargs):

        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = np.array(value)
            elif is_number(value):
                kwargs[key] = np.array(value).reshape((-1,))
        self.name = name
        self.syn = synmach(name, use_coi=use_coi, **kwargs)

        self.A = kwargs.get('A')
        self.B = kwargs.get('B')
        self.C = kwargs.get('C')
        self.D = kwargs.get('D')
        self.E = kwargs.get('E')
        self.W = kwargs.get('W')
        self.kp = kwargs.get('kp')
        self.ki = kwargs.get('ki')
        self.K1 = kwargs.get('K1')
        self.K2 = kwargs.get('K2')
        self.TRbase = kwargs.get('TRbase')
        self.wref = kwargs.get('wref')
        self.qmin = kwargs.get('qmin')
        self.qmax = kwargs.get('qmax')
        self.T1 = kwargs.get('T1')
        self.T2 = kwargs.get('T2')
        self.TCD = kwargs.get('TCD')
        self.TG = kwargs.get('TG')
        self.b = kwargs.get('b')
        self.TFS = kwargs.get('TFS')

        self.Tmec = None
        self.kNL = None
        self.qT_i = None
        self.qT = None
        self.Tref = None
        self.Tx = None
        self.Tr = None
        self.Tri = None
        self.Te = None
        self.qR = None
        self.qfuel = None
        self.xv = None
        self.qf = None
        self.Cop = None

    def gt_init(self):
        self.syn.synmach_init()
        self.Tmec = self.syn.Pm / self.syn.omega
        self.Cop = (self.Tmec - self.A - self.C * (1 - self.syn.omega)) / self.B
        self.qf = self.Cop
        self.xv = self.qf
        self.qfuel = self.xv
        self.qR = self.W * (self.wref - self.syn.omega)
        self.Te = self.TRbase + self.D * (1 - self.qf) + self.E * (1 - self.syn.omega)
        self.Tri = self.K2 * self.Te
        self.Tr = self.K1 * self.Te + self.Tri
        self.Tx = self.Tr
        self.Tref = self.Tx
        self.qT = self.qR
        self.qT_i = (self.qT - self.kp * (self.Tref - self.Tx)) / self.ki

        q = np.minimum(self.qT, np.clip(self.qR, self.qmin, self.qmax))
        self.kNL = (q - self.qfuel) / (q - 1)

    def mdl(self):
        self.gt_init()

        m = Model()
        name = self.name

        synmdl = self.syn.mdl(False)
        m.__dict__.update(synmdl.__dict__)

        # Exhaust temperature
        m.Te = Var('Te_' + name, value=self.Te)
        m.qf = Var('qf_' + name, value=self.qf)
        m.D = Param('D_' + name, value=self.D)
        m.E = Param('E_' + name, value=self.E)
        m.TRbase = Param('TRbase_' + name, value=self.TRbase)
        m.ExhTemp = Eqn('Exhaust Temperature', m.Te - (m.TRbase + m.D * (1 - m.qf) + m.E * (1 - m.omega)))

        # Radiation shield
        m.K1 = Param('K1_' + name, self.K1)
        m.K2 = Param('K2_' + name, self.K2)
        m.T1 = Param('T1_' + name, self.T1)

        m.Tr = Var('Tr_' + name, value=self.Tr)
        m.Tri = Var('Tri_' + name, self.Tri)
        m.RS1 = Ode('Radiation Shield1', f=(m.K2 * m.Te - m.Tri) / m.T1, diff_var=m.Tri)
        m.RS2 = Eqn('Radiation Shield2', m.Tr - (m.K1 * m.Te + m.Tri))

        # Thermocouple
        m.T2 = Param('T2_' + name, self.T2)
        m.Tx = Var('Tx_' + name, value=self.Tx)
        m.TC = Ode('Thermocouple_' + name, f=(m.Tr - m.Tx) / m.T2, diff_var=m.Tx)

        # Thermal controller
        m.kp = Param('kp_' + name, value=self.kp)
        m.ki = Param('ki_' + name, value=self.ki)
        m.qT_i = Var('qT_i_' + name, value=self.qT_i)
        m.qT = Var('qT_' + name, self.qT)
        m.Tref = Param('Tref_' + name, self.Tref)
        m.qmin = Param('qmin_' + name, self.qmin)
        m.qmax = Param('qmax_' + name, self.qmax)
        m.TC1 = Ode('Temp controller1', f=AntiWindUp(m.qT, m.qmin, m.qmax, m.Tref - m.Tx), diff_var=m.qT_i)
        m.TC2 = Eqn('Temp controller2', m.qT - (m.ki * m.qT_i + m.kp * (m.Tref - m.Tx)))

        # Compressor
        m.Cop = Var('Cop_' + name, value=self.Cop)
        m.TCD = Param('TCD_' + name, self.TCD)
        m.CompDisc = Ode('Compressor discharge', (m.qf - m.Cop) / m.TCD, diff_var=m.Cop)

        # Tmec
        m.Tmec = Var('Tmec_' + name, self.Tmec)
        m.A = Param('A_' + name, self.A)
        m.B = Param('B_' + name, self.B)
        m.C = Param('C_' + name, self.C)
        m.TorqMec = Eqn('Torque mechanic', m.Tmec - (m.A + m.B * m.Cop + m.C * (1 - m.omega)))
        m.gt_pm = Eqn('Mechanical power', m.Pm - m.Tmec * m.omega)

        # speed governor
        m.qR = Var('qR_' + name, self.qR)
        m.W = Param('W_' + name, self.W)
        m.TG = Param('TG_' + name, self.TG)
        m.wref = Param('wref_' + name, self.wref)
        m.SpeedGov = Ode('Speed governor', f=(m.W * (m.wref - m.omega) - m.qR) / m.TG, diff_var=m.qR)

        # fuel
        m.kNL = Param('kNL_' + name, self.kNL)
        m.qfuel = Var('qfuel_' + name, self.qfuel)
        rhs = m.qfuel - (m.kNL + (1 - m.kNL) * Min(m.qT, Saturation(m.qR, m.qmin, m.qmax)) * m.omega)
        m.FuelCons = Eqn('Fuel Consumption', rhs)

        # Valve Positioner
        m.xv = Var('xv_' + name, self.xv)
        m.b = Param('b_' + name, self.b)
        m.ValPos = Ode('Valve Positioner', (m.qfuel - m.xv) / m.b, diff_var=m.xv)
        m.TFS = Param('TFS_' + name, self.TFS)
        m.FuelDyn = Ode('Fuel Dynamics', (m.xv - m.qf) / m.TFS, diff_var=m.qf)

        m = rename_mdl(m, name)

        return m




