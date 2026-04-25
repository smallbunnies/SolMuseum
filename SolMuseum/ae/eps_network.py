import numpy as np
from scipy.sparse import csc_array
from Solverz import Eqn, Param, Model
from Solverz import Var, Abs, cos, sin
from Solverz import Idx, LoopEqn, Sum, TimeSeriesParam, Set
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
            sub-functions). Honored for both ``dyn=True`` and
            ``dyn=False``; pass ``loopeqn=False`` to fall back to
            the legacy per-bus scalar expansion.

        Notes
        -----
        Per-bus shunt admittance (dyn=True only)
            The ``dyn=True`` branch emits two ``TimeSeriesParam`` s
            at build time carrying a zero default profile,

                m.G_shunt = TimeSeriesParam(
                    'G_shunt',
                    v_series=[0.0, 0.0], time_series=[0.0, 1e9],
                    index=0, value=np.zeros(nb),
                )
                m.B_shunt = TimeSeriesParam(
                    'B_shunt',
                    v_series=[0.0, 0.0], time_series=[0.0, 1e9],
                    index=0, value=np.zeros(nb),
                )

            wired into the rectangular current-injection equations
            as an additional bus-to-ground admittance::

                ix[i] = Σ_j (G[i,j]·u_x[j] − B[i,j]·u_y[j])
                        − G_shunt[i]·u_x[i] + B_shunt[i]·u_y[i]
                iy[i] = Σ_j (G[i,j]·u_y[j] + B[i,j]·u_x[j])
                        − G_shunt[i]·u_y[i] − B_shunt[i]·u_x[i]

            so that the effective admittance seen at bus ``i`` is
            ``Y[i,i] + (G_shunt[i] + j·B_shunt[i])``. The default
            profile returns a length-``nb`` zero vector at every
            time step, so modules rendered without fault-injection
            intent are numerically identical to the pre-shunt
            version — the only build-time difference is that F_ / J_
            now evaluate ``G_shunt = p_["G_shunt"].get_v_t(t)`` once
            per step (a cheap ``scipy.interpolate.interp1d`` call
            plus an ``np.ndarray.copy()`` of length ``nb``).

            The reason ``G_shunt`` / ``B_shunt`` are declared as
            ``TimeSeriesParam`` rather than plain ``Param``: the code
            generator (``Solverz/code_printer/python/utilities.py``
            ``print_param``) dispatches on Param *class*: for
            ``TimeSeriesParam`` it emits
            ``name = p_["name"].get_v_t(t)``; for plain ``Param`` it
            emits ``name = p_["name"]`` — a raw dict lookup that
            hands the Param object through to a numba-compiled
            inner, which cannot type it. Declaring as
            ``TimeSeriesParam`` at build time is therefore required
            for the runtime swap pattern to work.

            Fault injection is done **at runtime** by replacing the
            zero-profile ``TimeSeriesParam`` on the compiled
            module's ``mdl.p`` dict with one carrying the actual
            fault profile. Typical 3-phase bolted-fault pattern at
            bus ``k`` over ``[t1, t2]``::

                from Solverz import TimeSeriesParam
                nom = np.zeros(pf.nb)
                mdl.p["G_shunt"] = TimeSeriesParam(
                    "G_shunt",
                    v_series=[0.0, 0.0, 1e6, 1e6, 0.0, 0.0],
                    time_series=[0, t1, t1 + 1e-3, t2, t2 + 1e-3, t_end],
                    index=k,
                    value=nom,
                )

            The ``index=k`` argument targets a single entry of the
            length-``nb`` vector: at each time step
            ``TimeSeriesParam.get_v_t`` starts from ``value.copy()``
            and overwrites ``value[k]`` with the interpolated
            scalar, so every other bus's shunt stays 0. Use
            ``B_shunt`` the same way for reactive-shunt faults;
            leave either Param untouched (keeps its zero default)
            for ordinary non-fault runs.
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

            if loopeqn:
                # LoopEqn path: ``Vm`` / ``Va`` are flat over every
                # bus (length ``nb``) instead of the legacy ``Vm[pq]``
                # / ``Va[pv+pq]`` subset Vars, but the exported names
                # stay as ``m.Vm`` / ``m.Va`` so downstream IES code
                # that references them by name keeps working. Ref+pv
                # Vm and ref Va are held at their known PF values via
                # two additional pin ``LoopEqn``s — no per-index
                # scalar Eqns.
                m.Vm = Var('Vm', np.asarray(Vm).copy())
                m.Va = Var('Va', np.asarray(Va).copy())
                m.Pg = Var('Pg', Pg)
                m.Qg = Var('Qg', Qg)
                m.Pd = Var('Pd', Pd)
                m.Qd = Var('Qd', Qd)
                m.Gbus = Param('Gbus', csc_array(G), dim=2, sparse=True)
                m.Bbus = Param('Bbus', csc_array(B), dim=2, sparse=True)

                i = Idx('i', nb)
                j = Idx('j', nb)
                body_P = (
                    m.Vm[i] * Sum(
                        m.Vm[j] * m.Gbus[i, j]
                        * cos(m.Va[i] - m.Va[j]),
                        j,
                    )
                    + m.Vm[i] * Sum(
                        m.Vm[j] * m.Bbus[i, j]
                        * sin(m.Va[i] - m.Va[j]),
                        j,
                    )
                    + m.Pd[i] - m.Pg[i]
                )
                m.P_eqn = LoopEqn('P_eqn', outer_index=i,
                                   body=body_P, model=m)

                body_Q = (
                    m.Vm[i] * Sum(
                        m.Vm[j] * m.Gbus[i, j]
                        * sin(m.Va[i] - m.Va[j]),
                        j,
                    )
                    - m.Vm[i] * Sum(
                        m.Vm[j] * m.Bbus[i, j]
                        * cos(m.Va[i] - m.Va[j]),
                        j,
                    )
                    + m.Qd[i] - m.Qg[i]
                )
                m.Q_eqn = LoopEqn('Q_eqn', outer_index=i,
                                   body=body_Q, model=m)

                # LoopEqn pins over (ref+pv, ref) subsets via
                # indirect-outer indexing, so the generated module
                # gets ONE inner_F per pin family instead of
                # ``nref + npv`` per-index scalar sub-functions.
                ref_pv_arr = np.array(ref + pv, dtype=int)
                ref_arr = np.array(ref, dtype=int)
                nref_pv = len(ref_pv_arr)
                nref = len(ref_arr)

                m.ref_pv_idx = Set('ref_pv_idx', ref_pv_arr)
                m.Vm_pinned = Param('Vm_pinned',
                                     np.asarray(Vm)[ref_pv_arr])
                m.ref_idx = Set('ref_idx', ref_arr)
                m.Va_pinned = Param('Va_pinned',
                                     np.asarray(Va)[ref_arr])

                i_vp = m.ref_pv_idx.idx('i_vp')
                i_vr = m.ref_idx.idx('i_vr')
                m.Vm_pin = LoopEqn(
                    'Vm_pin', outer_index=i_vp,
                    body=m.Vm[i_vp] - m.Vm_pinned[i_vp],
                    model=m,
                )
                m.Va_pin = LoopEqn(
                    'Va_pin', outer_index=i_vr,
                    body=m.Va[i_vr] - m.Va_pinned[i_vr],
                    model=m,
                )
            else:
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

            # Declared as ``TimeSeriesParam`` (not plain ``Param``) so
            # the code generator emits ``G_shunt = p_["G_shunt"].get_v_t(t)``
            # in the rendered ``F_`` / ``J_`` wrappers — see
            # ``Solverz/code_printer/python/utilities.py::print_param``.
            # The default profile is the length-``nb`` zero vector held
            # constant over ``[0, 1e9]`` s, so a module rendered without
            # fault-injection intent produces numerically identical F, J
            # to a pre-shunt module. Runtime fault injection is done by
            # replacing either Param with another ``TimeSeriesParam``
            # whose ``index`` / ``v_series`` / ``time_series`` encode
            # the faulted bus and the fault window — see the Notes block
            # of ``mdl()``'s docstring below.
            m.G_shunt = TimeSeriesParam(
                'G_shunt',
                v_series=[0.0, 0.0],
                time_series=[0.0, 1.0e9],
                index=0,
                value=np.zeros(nb),
            )
            m.B_shunt = TimeSeriesParam(
                'B_shunt',
                v_series=[0.0, 0.0],
                time_series=[0.0, 1.0e9],
                index=0,
                value=np.zeros(nb),
            )

            if loopeqn:
                m.Gbus = Param('Gbus', csc_array(self.pf.Gbus), dim=2, sparse=True)
                m.Bbus = Param('Bbus', csc_array(self.pf.Bbus), dim=2, sparse=True)

                i = Idx('i', nb)
                j = Idx('j', nb)
                body_ix = (
                    m.ix[i]
                    - Sum(m.Gbus[i, j] * m.ux[j], j)
                    + Sum(m.Bbus[i, j] * m.uy[j], j)
                    - m.G_shunt[i] * m.ux[i]
                    + m.B_shunt[i] * m.uy[i]
                )
                body_iy = (
                    m.iy[i]
                    - Sum(m.Gbus[i, j] * m.uy[j], j)
                    - Sum(m.Bbus[i, j] * m.ux[j], j)
                    - m.G_shunt[i] * m.uy[i]
                    - m.B_shunt[i] * m.ux[i]
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

                    rhs1 = rhs1 - m.G_shunt[i] * m.ux[i] + m.B_shunt[i] * m.uy[i]
                    rhs2 = rhs2 - m.G_shunt[i] * m.uy[i] - m.B_shunt[i] * m.ux[i]

                    m.__dict__[f'ix_inj_{i}'] = Eqn(f'ix injection {i}', rhs1)
                    m.__dict__[f'iy_inj_{i}'] = Eqn(f'iy injection {i}', rhs2)

        return m
