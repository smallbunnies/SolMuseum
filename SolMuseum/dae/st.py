from Solverz import Eqn, Ode, AliasVar, TimeSeriesParam, Param, Model, Saturation, AntiWindUp, Min
from Solverz import Var, Abs
from Solverz.sym_algebra.symbols import iVar, idx
from Solverz.utilities.type_checker import is_integer, is_number
from sympy import re as real, im as imag
import numpy as np
from warnings import warn

from .synmach import synmach
from .util import rename_mdl


class st:

    def __init__(self,
                 name='st',
                 use_coi=True,
                 **kwargs):

        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = np.array(value)
            elif is_number(value):
                kwargs[key] = np.array(value).reshape((-1,))
        self.name = name
        self.syn = synmach(name, use_coi=use_coi, **kwargs)

        self.phi = kwargs.get('phi')
        self.z = kwargs.get('z')
        self.F = kwargs.get('F')
        self.eta = kwargs.get('eta')
        self.TREF = kwargs.get('TREF')
        self.alpha = kwargs.get('alpha')
        self.mu_min = kwargs.get('mu_min')
        self.mu_max = kwargs.get('mu_max')
        self.TCH = kwargs.get('TCH')
        self.TRH = kwargs.get('TRH')
        self.kp = kwargs.get('kp')
        self.ki = kwargs.get('ki')

        self.mu1 = None
        self.Ts = None
        self.mu = None
        self.x2 = None
        self.x1 = None
        self.Pm = None

    def st_init(self):
        # heat to power
        # st initialization first then st

        self.Pm = self.eta*self.F-self.z*self.phi
        self.x1 = self.Pm
        self.x2 = self.Pm - self.alpha*self.x1
        self.mu = self.x1
        self.Ts = self.TREF
        self.mu1 = self.mu/self.ki

        self.syn.synmach_init()

        if not np.all(np.isclose(self.Pm, self.syn.Pm, atol=1e-6, rtol=1e-4)):
            warn(f"Pm output and grid injection not match, with deviation {np.abs(self.Pm-self.syn.Pm)}. "
                 "Please calculate ST output and perform the power flow first!")

    def mdl(self):
        self.st_init()

        m = Model()
        name = self.name

        synmdl = self.syn.mdl(False)
        m.__dict__.update(synmdl.__dict__)

        # volumn
        m.x1 = Var('x1_'+name, self.x1)
        m.x2 = Var('x2_'+name, self.x2)
        m.mu = Var('mu_'+name, self.mu)
        m.mu_min = Param('mu_min_'+name, self.mu_min)
        m.mu_max = Param('mu_max_'+name, self.mu_max)
        m.TCH = Param('TCH_'+name, self.TCH)
        m.alpha = Param('alpha_'+name, self.alpha)
        m.TRH = Param('TRH_'+name, self.TRH)
        m.volumn1 = Ode('volumn1_'+name, (Saturation(m.mu, m.mu_min, m.mu_max) - m.x1) / m.TCH, m.x1)
        m.volumn2 = Ode('volumn2_'+name, (m.x1 * (1 - m.alpha) - m.x2) / m.TRH, m.x2)
        m.eqn_Pm = Eqn('eqn_Pm', m.Pm - m.alpha * m.x1 - m.x2)

        # CHP
        m.z = Param('z_'+name, self.z)
        m.eta = Param('eta_'+name, self.eta)
        m.F = Param('F_'+name, self.F)
        m.phi = Var('phi_'+name, self.phi)
        m.chp = Eqn('chp', m.eta*m.F - m.Pm - m.z*m.phi)

        # temperature control
        m.ki = Param('ki_'+name, self.ki)
        m.kp = Param('kp_'+name, self.kp)
        m.TREF = Param('TREF_'+name, self.TREF)
        m.mu1 = Var('mu1_'+name, self.mu1)
        m.Ts = Var('Ts_'+name, self.Ts)
        m.temp_control1 = Ode('temp_control1_'+name,
                              AntiWindUp(m.mu, m.mu_min, m.mu_max, -(m.TREF - m.Ts)),
                              m.mu1)
        m.temp_control2 = Eqn('temp_control2', m.kp * (m.TREF - m.Ts) + m.ki * m.mu1 - m.mu)

        m = rename_mdl(m, name)

        return m
