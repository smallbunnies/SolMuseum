import numpy as np
from scipy.sparse import csc_array
from Solverz import Eqn, Param, Model, TimeSeriesParam, Var, Abs, heaviside, exp, Sign
from Solverz import Idx, LoopEqn, Sum
from Solverz.utilities.type_checker import is_number
from SolUtil import DhsFlow, DhsFaultFlow

from SolMuseum.pde import heat_pipe
from warnings import warn


class heat_network:

    def __init__(self, df: DhsFlow):
        warn("Apply only to prescribed mass flow directions only!")
        self.df = df
        if not self.df.run_succeed:
            self.df.run()

    def mdl(self,
            dx,
            dt=0,
            method='kt2',
            dynamic_slack=False,
            loopeqn=True):
        """Build the heat-network DAE model.

        When ``loopeqn=True`` (the default) the routine flattens the
        per-pipe ``Tsp_{j}`` / ``Trp_{j}`` temperature Vars into two
        contiguous ``Tsp_all`` / ``Trp_all`` Vars and expresses every
        per-node or per-pipe pattern (mass continuity, inlet-temp
        BCs, supply / return temperature mixing) as a single
        ``LoopEqn``. ``loopeqn=False`` keeps the original per-pipe
        scalar expansion byte-for-byte — useful both for regression
        comparisons and for users that still rely on the per-pipe
        ``Tsp_{j}`` Var names on the solution side.
        """
        if loopeqn:
            return self._mdl_loopeqn(dx, dt, method, dynamic_slack)
        return self._mdl_legacy(dx, dt, method, dynamic_slack)

    # ------------------------------------------------------------------
    # Legacy per-pipe Var expansion (original heat_network path).
    # ------------------------------------------------------------------
    def _mdl_legacy(self, dx, dt, method, dynamic_slack):
        m = Model()
        Tamb = self.df.Ta
        Cp = 4182
        m.Tamb = Param('Tamb', Tamb)
        m.m = Var('m', self.df.m)
        m.Ts = Var('Ts', self.df.Ts)
        m.Tr = Var('Tr', self.df.Tr)
        m.min = Var('min', self.df.minset)
        m.Tsource = Param('Tsource', self.df.hc['Ts'])
        m.Tload = Param('Tload', self.df.hc['Tr'])
        if dynamic_slack:
            m.Ts_slack = Var('Ts_slack', self.df.Ts[self.df.slack_node])
        m.lam_heat_pipe = Param('lam_heat_pipe', self.df.lam)
        m.Cp = Param('Cp', Cp)
        m.phi = Var('phi', self.df.phi)
        m.rho = Param('rho', 958.4)
        m.S = Param('S', self.df.S)

        L = self.df.L
        dx = dx
        M = np.floor(L / dx).astype(int)
        for j in range(self.df.n_pipe):
            attenuation = np.exp(- self.df.lam[j] * self.df.L[j] / (Cp * np.abs(self.df.m[j])))

            # supply pipe
            Tstart = self.df.Ts[self.df.pipe_from[j]]
            Tend = (Tstart - Tamb[0]) * attenuation + Tamb[0]
            Tsp0 = np.linspace(Tstart,
                               Tend,
                               M[j] + 1)

            # return pipe
            Tstart = self.df.Tr[self.df.pipe_to[j]]
            Tend = (Tstart - Tamb[0]) * attenuation + Tamb[0]
            Trp0 = np.linspace(Tstart,
                               Tend,
                               M[j] + 1)

            m.__dict__['Tsp_' + str(j)] = Var('Tsp_' + str(j), value=Tsp0)
            m.__dict__['Trp_' + str(j)] = Var('Trp_' + str(j), value=Trp0)

        n_node = self.df.n_node
        n_pipe = self.df.n_pipe

        # Per-node scalar mass continuity Eqns, BC Eqns emitted
        # inside the same loop (original legacy behaviour).
        for node in range(n_node):
            rhs = - m.min[node]
            for edge in self.df.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                rhs = rhs + m.m[pipe]
                idx = str(pipe)
                Trpj = m.__dict__['Trp_' + idx]
                m.__dict__[f'Return_pipe_inlet_temp_{pipe}'] = Eqn(
                    f'Return_pipe_inlet_temp_{pipe}',
                    Trpj[0] - m.Tr[node])

            for edge in self.df.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                rhs = rhs - m.m[pipe]
                idx = str(pipe)
                Tspj = m.__dict__['Tsp_' + idx]
                m.__dict__[f'Supply_pipe_inlet_temp_{pipe}'] = Eqn(
                    f'Supply_pipe_inlet_temp_{pipe}',
                    Tspj[0] - m.Ts[node])
            m.__dict__[f"Mass_flow_continuity_{node}"] = Eqn(
                f"Mass_flow_continuity_{node}", rhs)

        # loop pressure
        rhs = 0
        if len(self.df.pinloop) > 0 and not np.all(self.df.pinloop == 0):
            m.K = Param('K', self.df.K)
            for i in range(self.df.n_pipe):
                rhs += m.K[i] * m.m[i] ** 2 * Sign(m.m[i]) * self.df.pinloop[i]
            m.loop_pressure = Eqn("loop_pressure", rhs)

        # Supply temperature
        for node in range(self.df.n_node):
            lhs = 0
            rhs = 0

            if node in self.df.s_node.tolist() + self.df.slack_node.tolist():
                lhs += Abs(m.min[node])
                if node in self.df.slack_node.tolist() and dynamic_slack:
                    rhs += m.Ts_slack * Abs(m.min[node])
                else:
                    rhs += m.Tsource[node] * Abs(m.min[node])

            for edge in self.df.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                idx = str(pipe)
                Toutsj = m.__dict__['Tsp_' + idx][M[pipe]]
                lhs += heaviside(m.m[pipe]) * Abs(m.m[pipe])
                rhs += heaviside(m.m[pipe]) * (Toutsj * Abs(m.m[pipe]))

            for edge in self.df.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                idx = str(pipe)
                Toutsj = m.__dict__['Tsp_' + idx][0]
                lhs += (1 - heaviside(m.m[pipe])) * Abs(m.m[pipe])
                rhs += (1 - heaviside(m.m[pipe])) * (Toutsj * Abs(m.m[pipe]))

            lhs *= m.Ts[node]

            m.__dict__[f"Ts_{node}"] = Eqn(f"Ts_{node}", lhs - rhs)

        # Return temperature
        for node in range(self.df.n_node):
            lhs = 0
            rhs = 0

            if node in self.df.l_node:
                lhs += Abs(m.min[node])
                rhs += m.Tload[node] * Abs(m.min[node])

            for edge in self.df.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                idx = str(pipe)
                Toutrj = m.__dict__['Trp_' + idx][M[pipe]]
                lhs += heaviside(m.m[pipe]) * Abs(m.m[pipe])
                rhs += heaviside(m.m[pipe]) * (Toutrj * Abs(m.m[pipe]))

            for edge in self.df.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                idx = str(pipe)
                Toutrj = m.__dict__['Trp_' + idx][0]
                lhs += (1 - heaviside(m.m[pipe])) * Abs(m.m[pipe])
                rhs += (1 - heaviside(m.m[pipe])) * (Toutrj * Abs(m.m[pipe]))

            lhs *= m.Tr[node]

            m.__dict__[f"Tr_{node}"] = Eqn(f"Tr_{node}", lhs - rhs)

        # Temperature drop
        for edge in self.df.G.edges(data=True):
            pipe = edge[2]['idx']
            Tspj = m.__dict__[f'Tsp_{pipe}']
            m.add(heat_pipe(Tspj,
                            m.m[pipe],
                            m.lam_heat_pipe[pipe],
                            m.rho,
                            m.Cp,
                            m.S[pipe],
                            m.Tamb,
                            dx,
                            dt,
                            M[pipe],
                            's' + str(pipe),
                            method=method))
            Trpj = m.__dict__[f'Trp_{pipe}']
            m.add(heat_pipe(Trpj,
                            m.m[pipe],
                            m.lam_heat_pipe[pipe],
                            m.rho,
                            m.Cp,
                            m.S[pipe],
                            m.Tamb,
                            dx,
                            dt,
                            M[pipe],
                            'r' + str(pipe),
                            method=method))

        # heat power
        for node in range(self.df.n_node):

            phi = m.phi[node]

            if node in self.df.slack_node.tolist():
                if dynamic_slack:
                    rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Ts_slack - m.Tr[node])
                else:
                    rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Tsource[node] - m.Tr[node])
            elif node in self.df.s_node.tolist():
                rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Tsource[node] - m.Tr[node])
            elif node in self.df.l_node:
                rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Ts[node] - m.Tload[node])
            elif node in self.df.I_node:
                rhs = m.min[node]

            m.__dict__[f'phi_{node}'] = Eqn(f"phi_{node}", rhs)

        return m

    # ------------------------------------------------------------------
    # LoopEqn flat-Var expansion (Phase 1.3 refactor).
    #
    # Flattens the per-pipe ``Tsp_{j}`` / ``Trp_{j}`` temperature
    # Vars into two contiguous ``Tsp_all`` / ``Trp_all`` Vars of
    # length ``sum_j (M[j] + 1)``. Each pipe occupies a segment
    # ``Tsp_all[offsets[j] : offsets[j+1]]``. This unlocks the
    # following LoopEqn ports that were previously blocked on the
    # per-pipe Var symbol changing with ``j``:
    #
    #   - Per-pipe inlet-temperature BCs (``Tsp_all[inlet[j]] =
    #     Ts[pipe_from[j]]``) become one LoopEqn over pipes.
    #   - Per-node supply / return temperature mixing bodies that
    #     reference ``Tsp_all[outlet[p]]`` / ``Tsp_all[inlet[p]]``
    #     under a ``Sum`` over pipes, with sparse ``V_in`` /
    #     ``V_out`` incidence walkers.
    #   - Heat-power balance per node, with slack / source / load /
    #     intermediate behaviour selected via per-node mask Params.
    #
    # ``heat_pipe`` accepts a ``T_offset`` kwarg; the offset-aware
    # slices into ``Tsp_all`` / ``Trp_all`` give the PDE stencil
    # exactly the same per-pipe algebra as before, so the kt2
    # discretization is unchanged. Setting ``T_offset`` default to
    # ``0`` keeps the legacy call sites working verbatim.
    # ------------------------------------------------------------------
    def _mdl_loopeqn(self, dx, dt, method, dynamic_slack):
        m = Model()
        Tamb = self.df.Ta
        Cp = 4182
        m.Tamb = Param('Tamb', Tamb)
        m.m = Var('m', self.df.m)
        m.Ts = Var('Ts', self.df.Ts)
        m.Tr = Var('Tr', self.df.Tr)
        m.min = Var('min', self.df.minset)
        m.Tsource = Param('Tsource', self.df.hc['Ts'])
        m.Tload = Param('Tload', self.df.hc['Tr'])
        if dynamic_slack:
            m.Ts_slack = Var('Ts_slack', self.df.Ts[self.df.slack_node])
        m.lam_heat_pipe = Param('lam_heat_pipe', self.df.lam)
        m.Cp = Param('Cp', Cp)
        m.phi = Var('phi', self.df.phi)
        m.rho = Param('rho', 958.4)
        m.S = Param('S', self.df.S)

        L = self.df.L
        M = np.floor(L / dx).astype(int)
        n_node = self.df.n_node
        n_pipe = self.df.n_pipe

        # --- offsets and flat initial profiles ---
        offsets = np.zeros(n_pipe + 1, dtype=int)
        for j in range(n_pipe):
            offsets[j + 1] = offsets[j] + int(M[j]) + 1
        total_len = int(offsets[-1])
        Tsp0_all = np.empty(total_len)
        Trp0_all = np.empty(total_len)
        for j in range(n_pipe):
            attenuation = np.exp(- self.df.lam[j] * self.df.L[j]
                                  / (Cp * np.abs(self.df.m[j])))
            # supply pipe initial profile
            Tstart = self.df.Ts[self.df.pipe_from[j]]
            Tend = (Tstart - Tamb[0]) * attenuation + Tamb[0]
            Tsp0_all[offsets[j]:offsets[j + 1]] = np.linspace(
                Tstart, Tend, int(M[j]) + 1)
            # return pipe initial profile
            Tstart = self.df.Tr[self.df.pipe_to[j]]
            Tend = (Tstart - Tamb[0]) * attenuation + Tamb[0]
            Trp0_all[offsets[j]:offsets[j + 1]] = np.linspace(
                Tstart, Tend, int(M[j]) + 1)

        m.Tsp_all = Var('Tsp_all', Tsp0_all)
        m.Trp_all = Var('Trp_all', Trp0_all)

        pipe_inlet_idx_arr = offsets[:-1].astype(np.int64)
        pipe_outlet_idx_arr = (offsets[:-1] + M).astype(np.int64)
        pipe_from_node_arr = np.asarray(self.df.pipe_from, dtype=np.int64)
        pipe_to_node_arr = np.asarray(self.df.pipe_to, dtype=np.int64)
        m.pipe_inlet_idx = Param('pipe_inlet_idx', pipe_inlet_idx_arr, dtype=int)
        m.pipe_outlet_idx = Param('pipe_outlet_idx', pipe_outlet_idx_arr, dtype=int)
        m.pipe_from_node = Param('pipe_from_node', pipe_from_node_arr, dtype=int)
        m.pipe_to_node = Param('pipe_to_node', pipe_to_node_arr, dtype=int)

        # --- sparse node×pipe incidence matrices ---
        V_in_arr = np.zeros((n_node, n_pipe))
        V_out_arr = np.zeros((n_node, n_pipe))
        for node in range(n_node):
            for edge in self.df.G.in_edges(node, data=True):
                V_in_arr[node, edge[2]['idx']] = 1.0
            for edge in self.df.G.out_edges(node, data=True):
                V_out_arr[node, edge[2]['idx']] = 1.0
        V_net_arr = V_in_arr - V_out_arr
        m.V_in = Param('V_in', csc_array(V_in_arr), dim=2, sparse=True)
        m.V_out = Param('V_out', csc_array(V_out_arr), dim=2, sparse=True)
        m.V_net = Param('V_net', csc_array(V_net_arr), dim=2, sparse=True)

        # --- node-type masks for Tr_mixing LoopEqn ---
        #
        # Ts_mixing and phi_balance still reference the scalar
        # ``Ts_slack`` Var under ``dynamic_slack=True``, which
        # triggers Solverz's "Matrix derivative of scalar variables
        # not supported" guard in JacBlock. Keeping them as legacy
        # per-node scalar Eqns sidesteps the issue for now; only
        # ``Tr_mixing`` (which never touches ``Ts_slack``) is
        # LoopEqn-ified here.
        l_list = np.asarray(self.df.l_node).tolist()

        def _mask(idx_set):
            a = np.zeros(n_node)
            for idx in idx_set:
                a[int(idx)] = 1.0
            return a

        is_load_arr = _mask(l_list)
        m.is_load = Param('is_load', is_load_arr)

        # Indices shared by every LoopEqn below.
        i_n = Idx('i_n', n_node)
        p_p = Idx('p_p', n_pipe)

        # --- mass continuity LoopEqn ---
        body_mass = -m.min[i_n] + Sum(m.V_net[i_n, p_p] * m.m[p_p], p_p)
        m.Mass_flow_continuity = LoopEqn(
            'Mass_flow_continuity',
            outer_index=i_n,
            body=body_mass,
            model=m,
        )

        # --- loop pressure (single scalar) ---
        if len(self.df.pinloop) > 0 and not np.all(self.df.pinloop == 0):
            m.K = Param('K', self.df.K)
            rhs = 0
            for i in range(self.df.n_pipe):
                rhs += m.K[i] * m.m[i] ** 2 * Sign(m.m[i]) * self.df.pinloop[i]
            m.loop_pressure = Eqn("loop_pressure", rhs)

        # --- per-pipe inlet temperature BC as LoopEqns ---
        # Supply: Tsp_all[pipe_inlet_idx[p]] = Ts[pipe_from_node[p]]
        m.Supply_pipe_inlet_temp = LoopEqn(
            'Supply_pipe_inlet_temp',
            outer_index=p_p,
            body=(m.Tsp_all[m.pipe_inlet_idx[p_p]]
                  - m.Ts[m.pipe_from_node[p_p]]),
            model=m,
        )
        # Return: Trp_all[pipe_inlet_idx[p]] = Tr[pipe_to_node[p]]
        m.Return_pipe_inlet_temp = LoopEqn(
            'Return_pipe_inlet_temp',
            outer_index=p_p,
            body=(m.Trp_all[m.pipe_inlet_idx[p_p]]
                  - m.Tr[m.pipe_to_node[p_p]]),
            model=m,
        )

        # --- supply temperature mixing: legacy scalar per-node Eqn.
        #     Under ``dynamic_slack=True`` the slack row references
        #     ``m.Ts_slack`` (a length-1 Var) which trips the
        #     "Matrix derivative of scalar variables not supported"
        #     guard in Solverz's JacBlock when the LoopEqn row is
        #     differentiated against it. Until that guard is relaxed,
        #     we keep Ts_mixing as per-node scalar Eqns. It still
        #     references the flat ``Tsp_all`` so the per-pipe Vars
        #     stay fully retired.
        M_arr = M
        for node in range(n_node):
            lhs = 0
            rhs = 0
            if node in self.df.s_node.tolist() + self.df.slack_node.tolist():
                lhs += Abs(m.min[node])
                if node in self.df.slack_node.tolist() and dynamic_slack:
                    rhs += m.Ts_slack * Abs(m.min[node])
                else:
                    rhs += m.Tsource[node] * Abs(m.min[node])
            for edge in self.df.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                outlet = int(offsets[pipe]) + int(M_arr[pipe])
                Toutsj = m.Tsp_all[outlet]
                lhs += heaviside(m.m[pipe]) * Abs(m.m[pipe])
                rhs += heaviside(m.m[pipe]) * (Toutsj * Abs(m.m[pipe]))
            for edge in self.df.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                inlet = int(offsets[pipe])
                Toutsj = m.Tsp_all[inlet]
                lhs += (1 - heaviside(m.m[pipe])) * Abs(m.m[pipe])
                rhs += (1 - heaviside(m.m[pipe])) * (Toutsj * Abs(m.m[pipe]))
            lhs *= m.Ts[node]
            m.__dict__[f"Ts_{node}"] = Eqn(f"Ts_{node}", lhs - rhs)

        # --- return temperature mixing LoopEqn ---
        body_Tr = (
            m.Tr[i_n] * (
                m.is_load[i_n] * Abs(m.min[i_n])
                + Sum(m.V_out[i_n, p_p] * heaviside(m.m[p_p]) * Abs(m.m[p_p]), p_p)
                + Sum(m.V_in[i_n, p_p] * (1 - heaviside(m.m[p_p])) * Abs(m.m[p_p]), p_p)
            )
            - m.is_load[i_n] * m.Tload[i_n] * Abs(m.min[i_n])
            - Sum(m.V_out[i_n, p_p] * heaviside(m.m[p_p]) * Abs(m.m[p_p])
                  * m.Trp_all[m.pipe_outlet_idx[p_p]], p_p)
            - Sum(m.V_in[i_n, p_p] * (1 - heaviside(m.m[p_p])) * Abs(m.m[p_p])
                  * m.Trp_all[m.pipe_inlet_idx[p_p]], p_p)
        )
        m.Tr_mixing = LoopEqn(
            'Tr_mixing', outer_index=i_n, body=body_Tr, model=m,
        )

        # --- temperature drop: PDE per pipe via classic ``heat_pipe``.
        #     Flat ``Tsp_all`` / ``Trp_all`` + per-pipe ``T_offset``
        #     so the kt2 / iu / yao stencils target the right
        #     segment without changing ``heat_pipe`` itself.
        #     (An experimental LoopOde-based wrapper
        #     ``heat_pipe_loop`` lives alongside ``heat_pipe.py`` —
        #     a session-closed prototype that symbolic-AD's too
        #     slowly on the kt2 ``minmod`` body for the IES scale
        #     to be production-viable; kept as a starting point
        #     for a future sparsity-pre-built rewrite.)
        for j in range(n_pipe):
            off_j = int(offsets[j])
            m.add(heat_pipe(m.Tsp_all, m.m[j], m.lam_heat_pipe[j],
                             m.rho, m.Cp, m.S[j], m.Tamb, dx, dt,
                             int(M[j]), 's' + str(j),
                             method=method, T_offset=off_j))
            m.add(heat_pipe(m.Trp_all, m.m[j], m.lam_heat_pipe[j],
                             m.rho, m.Cp, m.S[j], m.Tamb, dx, dt,
                             int(M[j]), 'r' + str(j),
                             method=method, T_offset=off_j))

        # --- heat power: legacy scalar per-node (same Ts_slack
        #     scalar-Var Jacobian limitation as Ts_mixing). ---
        for node in range(n_node):
            phi = m.phi[node]
            if node in self.df.slack_node.tolist():
                if dynamic_slack:
                    rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Ts_slack - m.Tr[node])
                else:
                    rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Tsource[node] - m.Tr[node])
            elif node in self.df.s_node.tolist():
                rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Tsource[node] - m.Tr[node])
            elif node in self.df.l_node:
                rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Ts[node] - m.Tload[node])
            elif node in self.df.I_node:
                rhs = m.min[node]
            m.__dict__[f'phi_{node}'] = Eqn(f"phi_{node}", rhs)

        return m


class fault_heat_network:

    def __init__(self,
                 dff: DhsFaultFlow):
        self.dff = dff

    def mdl(self,
            dx=None,
            dt=0,
            method='kt2',
            leakage_diameter=None):
        dff = self.dff

        m = Model()

        Tamb = dff.df.Ta

        m.ms = Var('ms', dff.ms)
        m.mr = Var('mr', dff.mr)
        m.K = Param('K', dff.HydraSup.c)
        m.m_leak = Var('m_leak', dff.yf0['m_leak'])

        if leakage_diameter is None:
            m.S_leak = Param('S_leak', dff.df.S[dff.fault_pipe])
        else:
            m.S_leak = Param('S_leak', np.pi*leakage_diameter**2/4)

        m.g = Param('g', 10)
        m.Hs = Var('Hs', dff.Hs)
        m.Hr = Var('Hr', dff.Hr)
        m.Hset_s = Param('Hset_s', dff.HydraSup.Hset[0])
        m.Hset_r = Param('Hset_r', dff.HydraSup.Hset[0] - dff.dH)
        m.fs_injection = Var('fs_injection', dff.HydraSup.f[-1] + dff.HydraRet.f[-1])
        m.phi = Var('phi', dff.df.phi)
        m.Ts = Var('Ts', dff.Ts)
        m.Tr = Var('Tr', dff.Tr)
        m.min = Var('min', dff.minset[:-1])
        m.Tsource = Param('Tsource', dff.mdl_full.p['Tsource'] + Tamb)
        m.Tload = Param('Tload', dff.mdl_full.p['Tload'] + Tamb)
        m.lam = Param('lam', dff.mdl_full.p['lam'])
        m.S = Param('S', np.append(dff.df.S, [dff.df.S[dff.fault_pipe]]))
        m.Ls = Param('Ls', dff.mdl_full.p['Ls'])
        m.Lr = Param('Lr', dff.mdl_full.p['Lr'])
        m.Cp = Param('Cp', 4182)
        m.Ts_slack = Var('Ts_slack', dff.df.Ts[dff.df.slack_node])
        m.rho = Param('rho', 958.4)

        lam = m.lam.value
        L = dff.mdl_full.p['Ls']
        Cp = m.Cp.value
        dx = dx
        M = np.floor(L / dx).astype(int)
        for edge in dff.HydraSup.G.edges(data=True):
            fnode = edge[0]
            tnode = edge[1]
            pipe = edge[2]['idx']
            if pipe != dff.df.n_pipe + 1:
                attenuation = np.exp(- lam[pipe] * L[pipe] / (Cp * np.abs(dff.ms[pipe])))

                # supply pipe
                Tstart = dff.Ts[fnode] - Tamb
                Tend = Tstart * attenuation
                Tsp0 = np.linspace(Tstart + Tamb,
                                   Tend + Tamb,
                                   M[pipe] + 1).reshape(-1)

                # return pipe
                Tstart = dff.Tr[tnode] - Tamb
                Tend = Tstart * attenuation
                Trp0 = np.linspace(Tstart + Tamb,
                                   Tend + Tamb,
                                   M[pipe] + 1).reshape(-1)

                m.__dict__['Tsp_' + str(pipe)] = Var('Tsp_' + str(pipe), value=Tsp0)
                m.__dict__['Trp_' + str(pipe)] = Var('Trp_' + str(pipe), value=Trp0)

        # Supply temperature
        for node in range(dff.df.n_node + 1):
            # skip the leak node
            lhs = 0
            rhs = 0

            if node in dff.df.s_node.tolist() + dff.df.slack_node.tolist():
                lhs += Abs(m.min[node])
                if node in dff.df.slack_node:
                    rhs += m.Ts_slack * Abs(m.min[node])
                else:
                    rhs += m.Tsource[node] * Abs(m.min[node])

            for edge in dff.HydraSup.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                idx = str(pipe)
                Toutsj = m.__dict__['Tsp_' + idx][M[pipe]]
                lhs += heaviside(m.ms[pipe]) * Abs(m.ms[pipe])  #
                rhs += heaviside(m.ms[pipe]) * (Toutsj * Abs(m.ms[pipe]))  #

            for edge in dff.HydraSup.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                if pipe != dff.df.n_pipe + 1:
                    idx = str(pipe)
                    Toutsj = m.__dict__['Tsp_' + idx][0]
                    lhs += (1 - heaviside(m.ms[pipe])) * Abs(m.ms[pipe])  #
                    rhs += (1 - heaviside(m.ms[pipe])) * (Toutsj * Abs(m.ms[pipe]))  #

            lhs *= m.Ts[node]

            m.__dict__[f"Ts_{node}"] = Eqn(f"Ts_{node}", lhs - rhs)

        # Return temperature
        for node in range(dff.df.n_node + 1):
            # skip the leak node
            lhs = 0
            rhs = 0

            if node in dff.df.l_node:
                lhs += Abs(m.min[node])
                rhs += m.Tload[node] * Abs(m.min[node])

            for edge in dff.HydraRet.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                idx = str(pipe)
                Toutrj = m.__dict__['Trp_' + idx][M[pipe]]
                lhs += heaviside(m.mr[pipe]) * Abs(m.mr[pipe])  #
                rhs += heaviside(m.mr[pipe]) * (Toutrj * Abs(m.mr[pipe]))  #

            for edge in dff.HydraRet.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                if pipe != dff.df.n_pipe + 1:
                    idx = str(pipe)
                    Toutrj = m.__dict__['Trp_' + idx][0]
                    lhs += (1 - heaviside(m.mr[pipe])) * Abs(m.mr[pipe])  #
                    rhs += (1 - heaviside(m.mr[pipe])) * (Toutrj * Abs(m.mr[pipe]))  #

            lhs *= m.Tr[node]

            m.__dict__[f"Tr_{node}"] = Eqn(f"Tr_{node}", lhs - rhs)

        # Temperature drop
        m.Tamb = Param('Tamb', Tamb)
        for edge in dff.HydraSup.G.edges(data=True):
            fnode = edge[0]
            tnode = edge[1]
            pipe = edge[2]['idx']
            if pipe != dff.df.n_pipe + 1:  # DISCARD THE PIPE FROM LEAKAGE TO THE ENVIRONMENT

                Tspj = m.__dict__[f'Tsp_{pipe}']
                m.__dict__[f'Supply_pipe_inlet_temp_{pipe}'] = Eqn(f'Supply_pipe_inlet_temp_{pipe}',
                                                                   Tspj[0] - m.Ts[fnode])
                if M[pipe] > 0:
                    m.add(heat_pipe(Tspj,
                                    m.ms[pipe],
                                    m.lam[pipe],
                                    m.rho,
                                    m.Cp,
                                    m.S[pipe],
                                    m.Tamb,
                                    dx,
                                    dt,
                                    M[pipe],
                                    's' + str(pipe),
                                    method=method))
                # else:
                #     m.add(Eqn(f'heat_pipe_s_{pipe}',
                #               Tspj[1] - Tspj[0]))

                Trpj = m.__dict__[f'Trp_{pipe}']
                m.__dict__[f'Return_pipe_inlet_temp_{pipe}'] = Eqn(f'Return_pipe_inlet_temp_{pipe}',
                                                                   Trpj[0] - m.Tr[tnode])
                if M[pipe] > 0:
                    m.add(heat_pipe(Trpj,
                                    m.mr[pipe],
                                    m.lam[pipe],
                                    m.rho,
                                    m.Cp,
                                    m.S[pipe],
                                    m.Tamb,
                                    dx,
                                    dt,
                                    M[pipe],
                                    'r' + str(pipe),
                                    method=method))
                # else:
                #     m.add(Eqn(f'heat_pipe_r_{pipe}',
                #               Trpj[1] - Trpj[0]))

        # mass flow continuity
        # supply
        for node in range(dff.df.n_node):
            if node in dff.df.slack_node.tolist() + dff.df.s_node.tolist():
                rhs = Abs(m.min[node])
            elif node in dff.df.l_node.tolist() + dff.df.I_node.tolist():
                rhs = -Abs(m.min[node])
            else:
                raise ValueError(f"Unknown Node type {node}")

            for edge in dff.HydraSup.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                rhs = rhs + m.ms[pipe]

            for edge in dff.HydraSup.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                rhs = rhs - m.ms[pipe]
                idx = str(pipe)

            m.__dict__[f"Mass_flow_continuity_sup_{node}"] = Eqn(f"Mass_flow_continuity_sup_{node}", rhs)

        rhs = m.ms[dff.fault_pipe] - (m.ms[dff.df.n_pipe] + m.m_leak[0])
        m.__dict__[f"Mass_flow_continuity_sup_{dff.df.n_node}"] = Eqn(f"Mass_flow_continuity_sup_{dff.df.n_node}",
                                                                      rhs)

        # return
        for node in range(dff.df.n_node):
            if node in dff.df.slack_node:
                rhs = - (m.min[node] - m.fs_injection)
            elif node in dff.df.s_node:
                rhs = - m.min[node]
            elif node in dff.df.l_node.tolist() + dff.df.I_node.tolist():
                rhs = m.min[node]
            else:
                raise ValueError(f"Unknown Node type {node}")

            for edge in dff.HydraRet.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                rhs = rhs + m.mr[pipe]

            for edge in dff.HydraRet.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                rhs = rhs - m.mr[pipe]
            m.__dict__[f"Mass_flow_continuity_ret_{node}"] = Eqn(f"Mass_flow_continuity_ret_{node}", rhs)

        rhs = m.mr[dff.df.n_pipe] - (m.mr[dff.fault_pipe] + m.m_leak[1])
        m.__dict__[f"Mass_flow_continuity_ret_{dff.df.n_node}"] = Eqn(f"Mass_flow_continuity_ret_{dff.df.n_node}",
                                                                      rhs)

        # pressure drop
        for edge in dff.HydraSup.G.edges(data=True):
            fnode = edge[0]
            tnode = edge[1]
            pipe = edge[2]['idx']
            if pipe != dff.df.n_pipe + 1:  # DISCARD THE PIPE FROM LEAKAGE TO THE ENVIRONMENT
                rhs = m.Hs[fnode] - m.Hs[tnode] - m.K[pipe] * m.ms[pipe] ** 2 * Sign(m.ms[pipe])
                m.__dict__[f"Hs_{pipe}"] = Eqn(f"Hs_{pipe}", rhs)

                rhs = m.Hr[tnode] - m.Hr[fnode] - m.K[pipe] * m.mr[pipe] ** 2 * Sign(m.mr[pipe])
                m.__dict__[f"Hr_{pipe}"] = Eqn(f"Hr_{pipe}", rhs)

        m.Hs_slack = Eqn(f"Hs_slack", m.Hs[dff.df.slack_node[0]] - m.Hset_s)
        m.Hr_slack = Eqn(f"Hr_slack", m.Hr[dff.df.slack_node[0]] - m.Hset_r)

        m.leak_rate = TimeSeriesParam("leak_rate",
                                      [0, 0],
                                      [0, 100])
        # leak mass flow
        if dff.fault_sys == 's':
            rhs_s = m.m_leak[0] - m.leak_rate * m.S_leak * (2 * m.g * m.Hs[dff.df.n_node]) ** (1 / 2)
            rhs_r = m.m_leak[1]
        elif dff.fault_sys == 'r':
            rhs_s = m.m_leak[0]
            rhs_r = m.m_leak[1] - m.leak_rate * m.S_leak * (2 * m.g * m.Hr[dff.df.n_node]) ** (1 / 2)
        else:
            raise ValueError("Unknown fault sys")
        m.leak_mass_flow_sup = Eqn("leak_mass_flow_sup", rhs_s)
        m.leak_mass_flow_ret = Eqn("leak_mass_flow_ret", rhs_r)

        # heat power
        for node in range(dff.df.n_node):
            phi = m.phi[node]

            if node in dff.df.s_node:
                rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Tsource[node] - m.Tr[node])
            elif node in dff.df.slack_node:
                rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Ts_slack - m.Tr[node])
            elif node in dff.df.l_node:
                rhs = phi - m.Cp / 1e6 * Abs(m.min[node]) * (m.Ts[node] - m.Tload[node])
            elif node in dff.df.I_node:
                rhs = m.min[node]

            m.__dict__[f'phi_{node}'] = Eqn(f"phi_{node}", rhs)

        return m
