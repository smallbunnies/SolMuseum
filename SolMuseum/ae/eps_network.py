import numpy as np
from scipy.sparse import csc_array
from Solverz import Eqn, Param, Model
from Solverz import Var, Abs, cos, sin
from Solverz import Idx, LoopEqn, Sum
from Solverz.utilities.type_checker import is_number
from SolUtil import PowerFlow
from warnings import warn
from ..util import rename_mdl


class eps_network:

    def __init__(self, pf: PowerFlow):
        self.pf = pf
        self.pf.run()

    def mdl(self, dyn=False, loopeqn=True, **kwargs):
        """Build the EPS network model.

        Parameters
        ----------
        dyn : bool, default False
            If True, use the rectangular current-balance formulation
            (for transient/DAE use). Otherwise build the polar P/Q
            balance for steady-state power flow.
        loopeqn : bool, default True
            If True, emit the per-bus injection equations as
            ``LoopEqn`` scalar templates (one ``inner_F`` / ``inner_J``
            per equation family instead of ``O(n_bus)`` scalar
            sub-functions). Currently honored only when ``dyn=True``;
            the ``dyn=False`` branch always uses the legacy scalar
            expansion.
        """
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

            nb = self.pf.nb

            if loopeqn:
                m.Gbus = Param('Gbus', csc_array(self.pf.Gbus), dim=2, sparse=True)
                m.Bbus = Param('Bbus', csc_array(self.pf.Bbus), dim=2, sparse=True)

                i = Idx('i', nb)
                j = Idx('j', nb)
                body_ix = (
                    m.ix[i]
                    - Sum(m.Gbus[i, j] * m.ux[j], j)
                    + Sum(m.Bbus[i, j] * m.uy[j], j)
                )
                body_iy = (
                    m.iy[i]
                    - Sum(m.Gbus[i, j] * m.uy[j], j)
                    - Sum(m.Bbus[i, j] * m.ux[j], j)
                )
                m.ix_inj = LoopEqn('ix_inj', outer_index=i,
                                   body=body_ix, model=m)
                m.iy_inj = LoopEqn('iy_inj', outer_index=i,
                                   body=body_iy, model=m)
            else:
                for i in range(nb):
                    rhs1 = m.ix[i]
                    rhs2 = m.iy[i]

                    for j in range(nb):
                        rhs1 = rhs1 - self.pf.Gbus[i, j] * m.ux[j] + self.pf.Bbus[i, j] * m.uy[j]
                        rhs2 = rhs2 - self.pf.Gbus[i, j] * m.uy[j] - self.pf.Bbus[i, j] * m.ux[j]

                    m.__dict__[f'ix_inj_{i}'] = Eqn(f'ix injection {i}', rhs1)
                    m.__dict__[f'iy_inj_{i}'] = Eqn(f'iy injection {i}', rhs2)

        return m
