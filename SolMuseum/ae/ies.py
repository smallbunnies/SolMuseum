import numpy as np
from Solverz import Eqn, Param, Model
from Solverz import Var, Abs
from Solverz.utilities.type_checker import is_number
from warnings import warn
from ..util import rename_mdl


class p2g:

    def __init__(self,
                 name='p2g',
                 **kwargs):
        self.name = name

        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = np.array(value)
            elif is_number(value):
                kwargs[key] = np.array(value).reshape((-1,))

        self.h = kwargs.get('h')
        self.eta = kwargs.get('eta')
        self.c = kwargs.get('c')
        self.p = kwargs.get('p')
        self.q = kwargs.get('q')
        self.epsbase = kwargs.get('epsbase')
        self.pd = kwargs.get('pd')

    def p2g_init(self):
        pd = np.abs(self.q)*self.c**2*self.h/(self.eta*self.p)/self.epsbase
        if self.pd is None:
            self.pd = pd

    def mdl(self,
            rename=True):
        self.p2g_init()

        m = Model()

        name = self.name

        m.q = Var('q_' + name, self.q)
        m.p = Var('p_' + name, self.p)
        m.eta = Param('eta_' + name, self.eta)
        m.h = Param('h_' + name, self.h)
        m.c = Param('c_' + name, self.c)
        m.epsbase = Param('epsbase_' + name, self.epsbase)
        m.pd = Var('pd_' + name, self.pd)

        # power unit: MW
        m.eqn = Eqn('eqn_' + name, m.pd * m.epsbase - Abs(m.q) * m.c ** 2 * m.h / (m.eta * m.p))

        if rename:
            m = rename_mdl(m, name)

        return m


class eb:

    def __init__(self,
                 name='eb',
                 **kwargs):
        self.name = name

        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = np.array(value)
            elif is_number(value):
                kwargs[key] = np.array(value).reshape((-1,))

        self.eta = kwargs.get('eta')
        self.vm0 = kwargs.get('vm0')
        self.phi = kwargs.get('phi')
        self.ux = kwargs.get('ux')
        self.uy = kwargs.get('uy')
        self.epsbase = kwargs.get('epsbase')
        self.pd = None
        self.pd0 = kwargs.get('pd0')

    def eb_init(self):
        self.pd = self.phi / self.eta / self.epsbase
        pd = self.pd0*(self.ux**2+self.uy**2)/self.vm0**2

        if not np.all(np.isclose(pd, self.pd, atol=1e-6, rtol=1e-4)):
            warn(f"Electric power calculated by heat power and electric power injection not match, "
                 f"with deviation {np.abs(self.pd-pd)}.")

    def mdl(self,
            rename=True):
        self.eb_init()

        m = Model()

        name = self.name

        m.ux = Var('ux_'+name, self.ux)
        m.uy = Var('uy_'+name, self.uy)
        m.pd0 = Param('pd0_'+name, self.pd0)
        m.vm0 = Param('vm0_'+name, self.vm0)
        m.epsbase = Param('epsbase_'+name, self.epsbase)
        m.pd = Var('pd_'+name, self.pd)
        m.eta = Param('eta_'+name, self.eta)
        m.phi = Var('phi_'+name, self.phi)

        # power unit: W
        m.eqn_pd_electric = Eqn('eqn_pd_electric_'+name, m.pd - m.pd0*(m.ux**2+m.uy**2)/m.vm0**2)
        m.eqn_phi_pd = Eqn('eqn_phi_pd_'+name, m.pd*m.eta*m.epsbase - m.phi)

        if rename:
            m = rename_mdl(m, name)

        return m
