import numpy as np
from Solverz import Eqn, Param, Model
from Solverz import Var, Abs, cos, sin
from Solverz.utilities.type_checker import is_number
from SolUtil import PowerFlow
from warnings import warn
from ..util import rename_mdl


class eps_network:

    def __init__(self, pf: PowerFlow):
        self.pf = pf
        self.pf.run()

    def mdl(self, dyn=False, **kwargs):
        U: np.ndarray = self.pf.Vm * np.exp(1j * self.pf.Va)
        S: np.ndarray = (self.pf.Pg - self.pf.Pd) + 1j * (self.pf.Qg - self.pf.Qd)
        I = (S / U).conjugate()

        m = Model()

        if not dyn:
            Vm = self.pf.Vm
            Va = self.pf.Va
            nb = self.pf.nb
            Ybus = self.pf.Ybus
            G = Ybus.real
            B = Ybus.imag
            ref = self.pf.idx_slack.tolist()
            pv = self.pf.idx_pv.tolist()
            pq = self.pf.idx_pq.tolist()
            Pg = self.pf.Pg
            Qg = self.pf.Qg
            Pd = self.pf.Pd
            Qd = self.pf.Qd

            m.Va = Var('Va', Va[pv + pq])
            m.Vm = Var('Vm', Vm[pq])
            m.Pg = Var('Pg', Pg)
            m.Qg = Var('Qg', Qg)
            m.Pd = Var('Pd', Pd)
            m.Qd = Var('Qd', Qd)

            def get_Vm(idx):
                if idx in ref + pv:
                    return Vm[idx]
                elif idx in pq:
                    return m.Vm[pq.index(idx)]

            def get_Va(idx):
                if idx in ref:
                    return Va[idx]
                elif idx in pv + pq:
                    return m.Va[(pv + pq).index(idx)]

            for i in pv + pq + ref:
                expr = 0
                Vmi = get_Vm(i)
                Vai = get_Va(i)
                for j in range(nb):
                    Vmj = get_Vm(j)
                    Vaj = get_Va(j)
                    expr += Vmi * Vmj * (G[i, j] * cos(Vai - Vaj) + B[i, j] * sin(Vai - Vaj))
                m.__dict__[f'P_eqn_{i}'] = Eqn(f'P_eqn_{i}', expr + m.Pd[i] - m.Pg[i])

            for i in pv + pq + ref:
                expr = 0
                Vmi = get_Vm(i)
                Vai = get_Va(i)
                for j in range(nb):
                    Vmj = get_Vm(j)
                    Vaj = get_Va(j)
                    expr += Vmi * Vmj * (G[i, j] * sin(Vai - Vaj) - B[i, j] * cos(Vai - Vaj))
                m.__dict__[f'Q_eqn_{i}'] = Eqn(f'Q_eqn_{i}', expr + m.Qd[i] - m.Qg[i])

        else:
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
