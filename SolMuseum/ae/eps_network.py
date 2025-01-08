import numpy as np
from Solverz import Eqn, Param, Model
from Solverz import Var, Abs
from Solverz.utilities.type_checker import is_number
from SolUtil import PowerFlow
from warnings import warn
from ..util import rename_mdl


class eps_network:

    def __init__(self, pf: PowerFlow):
        self.pf = pf
        self.pf.run()

    def mdl(self, **kwargs):
        U: np.ndarray = self.pf.Vm * np.exp(1j * self.pf.Va)
        S: np.ndarray = (self.pf.Pg - self.pf.Pd) + 1j * (self.pf.Qg - self.pf.Qd)
        I = (S / U).conjugate()

        m = Model()
        m.ix = Var('ix', I.real)
        m.iy = Var('iy', I.imag)
        m.ux = Var('ux', U.real)
        m.uy = Var('uy', U.imag)
        m.Pd = Param('Pd', self.pf.Pd)
        m.Pg = Param('Pg', self.pf.Pg)
        m.Qd = Param('Qd', self.pf.Qd)
        m.Qg = Param('Qg', self.pf.Qg)

        # network mdl
        for i in range(self.pf.nb):
            rhs1 = m.ix[i]
            rhs2 = m.iy[i]

            for j in range(self.pf.nb):
                rhs1 = rhs1 - self.pf.Gbus[i, j] * m.ux[j] + self.pf.Bbus[i, j] * m.uy[j]
                rhs2 = rhs2 - self.pf.Gbus[i, j] * m.uy[j] - self.pf.Bbus[i, j] * m.ux[j]

            m.__dict__[f'ix_inj_{i}'] = Eqn(f'ix injection {i}', rhs1)
            m.__dict__[f'iy_inj_{i}'] = Eqn(f'iy injection {i}', rhs2)

        return m
