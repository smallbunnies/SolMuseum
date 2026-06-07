import sympy as sp
import numpy as np
from scipy.sparse import csc_array
from Solverz import Eqn, Param, Model, Idx, LoopEqn, LoopOde, Sum, TimeSeriesParam
from Solverz import Var, Abs
from Solverz import stamp_source
from Solverz.utilities.type_checker import is_number
from SolUtil import GasFlow

from SolMuseum.pde import ngs_pipe, leakage_pipe, rupture_pipe
from SolMuseum._version import __version__ as _sm_version


class gas_network:

    def __init__(self, gf: GasFlow):
        self.gf = gf
        self.gf.run()

    def mdl(self,
            dx,
            dt=0,
            method='weno3',
            fault_type=None,
            fault_pipe_index=[],
            fault_loc_index=[],
            leak_diameter=[],
            loopeqn=True,
            bundle_leak=False):
        if loopeqn:
            m = self._mdl_loopeqn(dx, dt, method,
                                  fault_type=fault_type,
                                  fault_pipe_index=fault_pipe_index,
                                  fault_loc_index=fault_loc_index,
                                  leak_diameter=leak_diameter,
                                  bundle_leak=bundle_leak)
        else:
            m = self._mdl_legacy(dx, dt, method, fault_type,
                                 fault_pipe_index, fault_loc_index,
                                 leak_diameter)
        stamp_source(m, component='gas_network', package='SolMuseum', version=_sm_version)
        return m

    # ------------------------------------------------------------------
    def _mdl_legacy(self, dx, dt, method, fault_type,
                    fault_pipe_index, fault_loc_index, leak_diameter):
        m = Model()
        m.Pi = Var('Pi', value=self.gf.Pi * 1e6)
        m.fs = Var('fs', value=self.gf.fs)
        m.fl = Var('fl', value=self.gf.fl)
        va = self.gf.va
        m.D = Param('D', value=self.gf.D)
        m.Area = Param('Area', np.pi * (self.gf.D / 2) ** 2)
        m.lam_gas_pipe = Param('lam_gas_pipe', value=self.gf.lam)
        L = self.gf.L
        M = np.floor(L / dx).astype(int)
        for j in range(self.gf.n_pipe):
            p0 = np.linspace(self.gf.Pi[self.gf.rpipe_from[j]]*1e6,
                             self.gf.Pi[self.gf.rpipe_to[j]]*1e6,
                             M[j] + 1)
            m.__dict__['p' + str(j)] = Var('p' + str(j), value=p0)
            m.__dict__['q' + str(j)] = Var('q' + str(j), value=self.gf.f[j] * np.ones(M[j] + 1))

        for node in self.gf.gc['G'].nodes:
            eqn_q = - m.fs[node] + m.fl[node]
            for edge in self.gf.gc['G'].in_edges(node, data=True):
                pipe = edge[2]['idx']
                idx = str(pipe)
                ptype = edge[2]['type']
                qi = m.__dict__['q' + idx]
                pi = m.__dict__['p' + idx]
                if ptype == 1:
                    eqn_q = eqn_q - qi[M[pipe]]
                    m.__dict__[f'pressure_outlet_pipe{idx}'] = Eqn(
                        f'Pressure node {node} pipe {idx} outlet',
                        m.Pi[node] - pi[M[pipe]])
            for edge in self.gf.gc['G'].out_edges(node, data=True):
                pipe = edge[2]['idx']
                idx = str(pipe)
                ptype = edge[2]['type']
                qi = m.__dict__['q' + idx]
                pi = m.__dict__['p' + idx]
                if ptype == 1:
                    eqn_q = eqn_q + qi[0]
                    m.__dict__[f'pressure_inlet_pipe{idx}'] = Eqn(
                        f'Pressure node {node} pipe {idx} inlet',
                        m.Pi[node] - pi[0])
            m.__dict__[f'mass_continuity_node{node}'] = Eqn(
                f'mass flow continuity of node {node}', eqn_q)

        for i in self.gf.slack:
            m.__dict__[f'node_pressure_{i}'] = Eqn(
                f'node_pressure_{i}',
                m.Pi[i] - self.gf.Piset[self.gf.slack.tolist().index(i)]*1e6)

        for edge in self.gf.gc['G'].edges(data=True):
            j = edge[2]['idx']
            Mj = M[j]
            pj = m.__dict__['p' + str(j)]
            qj = m.__dict__['q' + str(j)]
            if j not in fault_pipe_index:
                ngs_p = ngs_pipe(pj, qj, m.lam_gas_pipe[j], va,
                                 m.D[j], m.Area[j], dx, dt, Mj,
                                 f'pipe{j}', method=method)
            else:
                id_fault = fault_pipe_index.index(j)
                if fault_type == 'leakage':
                    ngs_p = leakage_pipe(
                        pj, qj, m.lam_gas_pipe[j], va, m.D[j],
                        m.Area[j], dx, dt, Mj, f'pipe{j}_leakage',
                        fault_loc_index[id_fault],
                        leak_diameter[id_fault], method=method)
                elif fault_type == 'rupture':
                    ngs_p = rupture_pipe(
                        pj, qj, m.lam_gas_pipe[j], va, m.D[j],
                        m.Area[j], dx, dt, Mj, f'pipe{j}_rupture',
                        fault_loc_index[id_fault], method=method)
                else:
                    raise ValueError(f'Fault type {fault_type} not supported!')
            for key, value in ngs_p.items():
                m.__dict__[key] = value
        return m

    # ------------------------------------------------------------------
    def _mdl_loopeqn(self, dx, dt, method,
                     fault_type=None,
                     fault_pipe_index=None,
                     fault_loc_index=None,
                     leak_diameter=None,
                     bundle_leak=False):
        """Build the gas-network model using LoopEqn / LoopOde.

        Method support
        --------------
        The bundled-state builder below inlines a WENO3 five-point
        interior stencil plus a TVD1 three-point boundary stencil. The
        ``method`` argument is accepted for signature compatibility
        with ``_mdl_legacy`` but only ``'weno3'`` is honoured. Pass
        any other scheme through ``mdl(loopeqn=False, method=...)``.

        Flat layout
        -----------
        ``p_all`` / ``q_all`` pack all pipes into one Var each:

            [ state_cells_pipe0 | ... | bc_inlet_cells | bc_outlet_cells
              | leak_alg_cells | qleak1_cells | qleak2_cells ]

        State cells per pipe: ``M[j]-1`` cells for an un-faulted pipe
        (original cells 1..M-1), or ``M[j]-2`` cells for a faulted
        pipe (original cells [1..il-1, il+1..M-1], i.e. the leak cell
        ``il`` is excluded from the state slice and packed at the
        tail as an algebraic variable).

        Tail layout, in order:

        * ``[total_state, total_state+n_pipe)`` — BC inlet cell of
          each pipe (original cell 0).
        * ``[total_state+n_pipe, total_state+2*n_pipe)`` — BC outlet
          cell of each pipe (original cell ``M[j]``).
        * ``[total_state+2*n_pipe, total_state+2*n_pipe+n_fault)`` —
          algebraic leak cell of each fault pipe (``p[il]`` in
          ``p_all``; ``q[il]`` in ``q_all``).
        * ``[total_state+2*n_pipe+n_fault, +2*n_fault)`` —
          ``qleak1`` of each fault pipe (only meaningful in
          ``q_all``; the corresponding entry in ``p_all`` is unused
          padding).
        * ``[total_state+2*n_pipe+2*n_fault, +3*n_fault)`` —
          ``qleak2`` of each fault pipe (only meaningful in
          ``q_all``).

        Equation structure
        ------------------
        * **mass_continuity** — LoopEqn over all nodes
        * **pressure_inlet / pressure_outlet** — LoopEqn over pipes
        * **slack_pressure** — scalar Eqn per slack node
        * **bd_right / bd_left** — LoopEqn over pipes (characteristic BC
          at cells 0 and M; equation form is the same for fault and
          no-fault pipes)
        * **leak_bd_left / leak_bd_right** — LoopEqn over fault pipes
          (characteristic BC at the leak cell; replaces the two
          ``leak3_1`` / ``leak3_2`` algebraic Eqns from the legacy
          ``leakage_ngs_pipe_weno3`` recipe)
        * **weno3_pipe{j}_leakage_bd1 / _bd2** — scalar Eqns per
          fault pipe (``q[il] = qjleak(p[il])`` and
          ``qleak1 - qleak2 - q[il] = 0``).  Scalar because each
          fault pipe carries its own ``leak_rate`` TimeSeriesParam.
        * **gas_pipe_q / gas_pipe_p** — cross-pipe LoopOde (weno3
          interior + TVD1 boundary via mask). For fault pipes the
          stencil neighbour positions are patched so that the TVD1
          cells immediately bracketing the leak read ``qleak1`` /
          ``qleak2`` instead of ``q[il]``; the leak cell itself is
          excluded from the LoopOde because ``q_all[leak]`` is now
          algebraic.
        """
        if fault_pipe_index is None:
            fault_pipe_index = []
        if fault_loc_index is None:
            fault_loc_index = []
        if leak_diameter is None:
            leak_diameter = []
        if method != 'weno3':
            raise NotImplementedError(
                f"gas_network.mdl(loopeqn=True) only supports method='weno3' "
                f"(got method={method!r}); the LoopEqn bundled-state builder "
                f"inlines a WENO3 stencil and no other scheme is wired through "
                f"it. Use mdl(loopeqn=False, method={method!r}) for the legacy "
                f"per-pipe discretizations (weno3 / kt1 / cdm / cha / euler).")
        if fault_type not in (None, 'leakage'):
            raise NotImplementedError(
                f'_mdl_loopeqn does not yet support fault_type={fault_type!r}; '
                f'use mdl(loopeqn=False) for rupture builds.')

        gf = self.gf
        n_node = gf.n_node
        n_pipe = gf.n_pipe
        va = gf.va
        L = gf.L
        M = np.floor(L / dx).astype(int)

        if (M < 2).any():
            bad = np.where(M < 2)[0]
            raise ValueError(
                f'gas_network.mdl(loopeqn=True) requires floor(L/dx) >= 2 for '
                f'every pipe so the WENO3 / TVD1 boundary stencils address '
                f'distinct state cells. {len(bad)} pipe(s) have M < 2 at '
                f'dx={dx}: indices {bad[:10].tolist()}'
                f'{"..." if len(bad) > 10 else ""}, '
                f'lengths {np.asarray(L)[bad][:10].tolist()}. A pipe with '
                f'M = floor(L/dx) <= 1 carries no interior state cell, so its '
                f'bd_left / bd_right characteristic-BC equations would collide '
                f'with a neighbouring pipe\'s state and inject O(MPa) initial '
                f'residuals. Reduce dx or remove/lengthen the short stub pipes '
                f'before building the dynamic model.')

        # type-0 (compressor) pipes cannot be represented in the dynamic
        # LoopEqn model. Only type-1 edges populate the mass-continuity
        # incidence Q_net, so a type-0 pipe's flow is silently dropped and
        # leaves a phantom residual at every compressor-adjacent node;
        # there is also no compressor pressure-boost term. Reject up front
        # instead of building a quietly-wrong model.
        edge_types = np.ones(n_pipe, dtype=np.int64)
        for edge in gf.gc['G'].edges(data=True):
            edge_types[edge[2]['idx']] = edge[2].get('type', 1)
        if (edge_types != 1).any():
            bad = np.where(edge_types != 1)[0]
            raise NotImplementedError(
                f'gas_network.mdl(loopeqn=True) only supports type-1 '
                f'(ordinary friction) pipes. {len(bad)} non-type-1 edge(s) '
                f'found: {bad[:10].tolist()}'
                f'{"..." if len(bad) > 10 else ""}. Type-0 (compressor) pipes '
                f'are dropped from the dynamic mass-continuity balance and '
                f'have no pressure-boost model, so they cannot be represented '
                f'here. Fold the compressor onto an adjacent type-1 pipe in '
                f'the steady-state data, or keep the compressor in the '
                f'SolUtil.GasFlow steady-state model only.')
        delta = np.asarray(getattr(gf, 'delta', np.zeros(n_pipe)))
        if delta.size and np.any(delta != 0):
            bad = np.where(delta != 0)[0]
            raise NotImplementedError(
                f'gas_network.mdl(loopeqn=True) cannot apply a compressor '
                f'pressure boost: {len(bad)} pipe(s) carry a non-zero delta '
                f'({bad[:10].tolist()}'
                f'{"..." if len(bad) > 10 else ""}). The dynamic LoopEqn '
                f'pressure_outlet condition enforces Pi[to] = p_all[outlet] '
                f'with no offset, so a non-zero delta would be ignored. Remove '
                f'the boost from the pipe data, or model the compressor in the '
                f'steady-state GasFlow only.')

        n_fault = len(fault_pipe_index)
        if n_fault > 0 and fault_type != 'leakage':
            raise ValueError(
                f'fault_pipe_index is non-empty but fault_type={fault_type!r}; '
                f'set fault_type=\'leakage\' to enable the LoopEqn fault path.')
        if not (len(fault_loc_index) == n_fault and len(leak_diameter) == n_fault):
            raise ValueError(
                'fault_pipe_index / fault_loc_index / leak_diameter must have '
                f'the same length (got {n_fault}, {len(fault_loc_index)}, '
                f'{len(leak_diameter)}).')

        is_fault = np.zeros(n_pipe, dtype=bool)
        fault_of = -np.ones(n_pipe, dtype=np.int64)
        leak_cell = np.zeros(n_pipe, dtype=np.int64)
        for fi, j in enumerate(fault_pipe_index):
            if not (2 <= int(fault_loc_index[fi]) <= int(M[j]) - 2):
                raise ValueError(
                    f'fault_loc_index[{fi}]={fault_loc_index[fi]} out of range '
                    f'[2, M[{j}]-2={int(M[j]) - 2}].')
            is_fault[j] = True
            fault_of[j] = fi
            leak_cell[j] = int(fault_loc_index[fi])

        m = Model()
        m.Pi = Var('Pi', value=gf.Pi * 1e6)
        m.fs = Var('fs', value=gf.fs)
        m.fl = Var('fl', value=gf.fl)
        m.D = Param('D', value=gf.D)
        m.Area = Param('Area', np.pi * (gf.D / 2) ** 2)
        m.lam_gas_pipe = Param('lam_gas_pipe', value=gf.lam)
        m.va = Param('va', va)

        # --- per-pipe state cell counts ---
        n_state_per = np.array(
            [int(M[j]) - (2 if is_fault[j] else 1) for j in range(n_pipe)],
            dtype=np.int64)
        state_offsets = np.concatenate(
            [[0], np.cumsum(n_state_per)]).astype(np.int64)
        total_state = int(state_offsets[-1])
        n_pipe_tail = 2 * n_pipe                  # bc_inlet + bc_outlet
        # p_all carries only the algebraic leak cell p[il] per fault
        # pipe; qleak1 / qleak2 live in q_all only, so p_all is
        # shorter than q_all by 2 * n_fault.
        p_total_len = total_state + n_pipe_tail + n_fault
        q_total_len = total_state + n_pipe_tail + 3 * n_fault

        bc_inlet_pos = (total_state + np.arange(n_pipe)).astype(np.int64)
        bc_outlet_pos = (total_state + n_pipe + np.arange(n_pipe)).astype(np.int64)
        # Per-pipe positions of the leak tail slots (indexed by *pipe*,
        # not by fault list; -1 for non-fault pipes, never read by any
        # Eqn). leak_alg_pipe is valid in both p_all and q_all; qleak1
        # / qleak2 are valid in q_all only.
        leak_alg_pipe = -np.ones(n_pipe, dtype=np.int64)
        qleak1_pipe = -np.ones(n_pipe, dtype=np.int64)
        qleak2_pipe = -np.ones(n_pipe, dtype=np.int64)
        for fi, j in enumerate(fault_pipe_index):
            leak_alg_pipe[j] = total_state + n_pipe_tail + fi
            qleak1_pipe[j] = total_state + n_pipe_tail + n_fault + fi
            qleak2_pipe[j] = total_state + n_pipe_tail + 2 * n_fault + fi

        # --- initial values ---
        p0_all = np.zeros(p_total_len)
        q0_all = np.zeros(q_total_len)
        for j in range(n_pipe):
            Mj = int(M[j])
            soff = int(state_offsets[j])
            p_profile = np.linspace(
                gf.Pi[gf.rpipe_from[j]] * 1e6,
                gf.Pi[gf.rpipe_to[j]] * 1e6, Mj + 1)
            if is_fault[j]:
                il = int(leak_cell[j])
                ns_left = il - 1
                ns_right = Mj - 1 - il
                if ns_left > 0:
                    p0_all[soff:soff + ns_left] = p_profile[1:il]
                    q0_all[soff:soff + ns_left] = gf.f[j]
                if ns_right > 0:
                    p0_all[soff + ns_left:soff + ns_left + ns_right] = p_profile[il + 1:Mj]
                    q0_all[soff + ns_left:soff + ns_left + ns_right] = gf.f[j]
                p0_all[leak_alg_pipe[j]] = p_profile[il]
                q0_all[leak_alg_pipe[j]] = gf.f[j]
                # qleak1 / qleak2 init = neighbouring state flows
                q0_all[qleak1_pipe[j]] = gf.f[j]
                q0_all[qleak2_pipe[j]] = gf.f[j]
            else:
                p0_all[soff:soff + Mj - 1] = p_profile[1:Mj]
                q0_all[soff:soff + Mj - 1] = gf.f[j]
            p0_all[total_state + j] = p_profile[0]
            p0_all[total_state + n_pipe + j] = p_profile[Mj]
            q0_all[total_state + j] = gf.f[j]
            q0_all[total_state + n_pipe + j] = gf.f[j]
        m.p_all = Var('p_all', p0_all)
        m.q_all = Var('q_all', q0_all)

        # --- pipe-level index Params ---
        m.bc_inlet_pos = Param('bc_inlet_pos', bc_inlet_pos, dtype=int)
        m.bc_outlet_pos = Param('bc_outlet_pos', bc_outlet_pos, dtype=int)

        pipe_from_node = np.empty(n_pipe, dtype=np.int64)
        pipe_to_node = np.empty(n_pipe, dtype=np.int64)
        for edge in gf.gc['G'].edges(data=True):
            j = edge[2]['idx']
            pipe_from_node[j] = edge[0]
            pipe_to_node[j] = edge[1]
        m.gas_pipe_from = Param('gas_pipe_from', pipe_from_node, dtype=int)
        m.gas_pipe_to = Param('gas_pipe_to', pipe_to_node, dtype=int)

        # --- sparse node x q_all incidence for mass continuity ---
        # Width matches q_all length; leak-tail columns are all zero,
        # which is the correct physics (leak gas leaves the network at
        # the leak boundary, not at a node).
        from scipy.sparse import lil_array as _lil
        Q_net = _lil((n_node, q_total_len))
        for node in gf.gc['G'].nodes:
            for edge in gf.gc['G'].in_edges(node, data=True):
                if edge[2]['type'] == 1:
                    Q_net[node, bc_outlet_pos[edge[2]['idx']]] = -1.0
            for edge in gf.gc['G'].out_edges(node, data=True):
                if edge[2]['type'] == 1:
                    Q_net[node, bc_inlet_pos[edge[2]['idx']]] = 1.0
        m.Q_net_gas = Param('Q_net_gas', csc_array(Q_net), dim=2, sparse=True)

        i_n = Idx('i_n', n_node)
        p_p = Idx('p_p', n_pipe)
        k_q = Idx('k_q', q_total_len)

        m.mass_continuity = LoopEqn(
            'mass_continuity', outer_index=i_n,
            body=(-m.fs[i_n] + m.fl[i_n]
                  + Sum(m.Q_net_gas[i_n, k_q] * m.q_all[k_q], k_q)),
            model=m)

        m.pressure_inlet = LoopEqn(
            'pressure_inlet', outer_index=p_p,
            body=m.Pi[m.gas_pipe_from[p_p]] - m.p_all[m.bc_inlet_pos[p_p]],
            model=m)
        m.pressure_outlet = LoopEqn(
            'pressure_outlet', outer_index=p_p,
            body=m.Pi[m.gas_pipe_to[p_p]] - m.p_all[m.bc_outlet_pos[p_p]],
            model=m)

        for i in gf.slack:
            m.__dict__[f'node_pressure_{i}'] = Eqn(
                f'node_pressure_{i}',
                m.Pi[i] - gf.Piset[gf.slack.tolist().index(i)] * 1e6)

        # --- characteristic BC at cells 0 / M (per-pipe LoopEqn) ---
        # State positions for cells 1, 2, M-2, M-1 are valid for both
        # fault and no-fault pipes because cells 1, 2 always live in
        # the upstream segment and cells M-2, M-1 in the downstream
        # segment; the leak (2 <= il <= M-2) never falls on any of
        # them (note: the il==M-2 case is rejected as out of range by
        # the input check above only when M-2 < 2, but for valid il
        # the bd_right stencil still resolves correctly via
        # contiguous packing of the downstream segment).
        first_state_pos = state_offsets[:-1].astype(np.int64)
        second_state_pos = (first_state_pos + 1).astype(np.int64)
        last_state_pos = (state_offsets[:-1] + n_state_per - 1).astype(np.int64)
        second_last_state_pos = (last_state_pos - 1).astype(np.int64)
        m.first_state = Param('first_state', first_state_pos, dtype=int)
        m.second_state = Param('second_state', second_state_pos, dtype=int)
        m.last_state = Param('last_state', last_state_pos, dtype=int)
        m.second_last_state = Param('second_last_state', second_last_state_pos, dtype=int)

        m.bd_right = LoopEqn(
            'bd_right', outer_index=p_p,
            body=(m.Area[p_p] * m.p_all[m.bc_outlet_pos[p_p]]
                  + m.va * m.q_all[m.bc_outlet_pos[p_p]]
                  + m.Area[p_p] * m.p_all[m.second_last_state[p_p]]
                  + m.va * m.q_all[m.second_last_state[p_p]]
                  - 2 * (m.Area[p_p] * m.p_all[m.last_state[p_p]]
                         + m.va * m.q_all[m.last_state[p_p]])),
            model=m)
        m.bd_left = LoopEqn(
            'bd_left', outer_index=p_p,
            body=(m.Area[p_p] * m.p_all[m.second_state[p_p]]
                  - m.va * m.q_all[m.second_state[p_p]]
                  + m.Area[p_p] * m.p_all[m.bc_inlet_pos[p_p]]
                  - m.va * m.q_all[m.bc_inlet_pos[p_p]]
                  - 2 * (m.Area[p_p] * m.p_all[m.first_state[p_p]]
                         - m.va * m.q_all[m.first_state[p_p]])),
            model=m)

        # --- fault-specific LoopEqns + per-fault scalar Eqns ---
        if n_fault > 0:
            fault_pipe_arr = np.array(fault_pipe_index, dtype=np.int64)
            leak_alg_fault = np.array(
                [leak_alg_pipe[j] for j in fault_pipe_arr], dtype=np.int64)
            qleak1_fault = np.array(
                [qleak1_pipe[j] for j in fault_pipe_arr], dtype=np.int64)
            qleak2_fault = np.array(
                [qleak2_pipe[j] for j in fault_pipe_arr], dtype=np.int64)
            # Positions of cells il-2, il-1, il+1, il+2 in the new state
            # layout (used by the leak characteristic BC LoopEqns).
            leak_lm2 = np.zeros(n_fault, dtype=np.int64)
            leak_lm1 = np.zeros(n_fault, dtype=np.int64)
            leak_rp1 = np.zeros(n_fault, dtype=np.int64)
            leak_rp2 = np.zeros(n_fault, dtype=np.int64)
            for fi, j in enumerate(fault_pipe_arr):
                soff = int(state_offsets[j])
                il = int(leak_cell[j])
                # Cells in upstream segment: local k = c - 1 for c in [1, il-1].
                leak_lm2[fi] = soff + (il - 3)      # cell il-2  (needs il >= 3)
                leak_lm1[fi] = soff + (il - 2)      # cell il-1
                # Cells in downstream segment: local k = c - 2 for c in [il+1, M-1].
                leak_rp1[fi] = soff + (il - 1)      # cell il+1
                leak_rp2[fi] = soff + il            # cell il+2  (needs il <= M-3)
            m.fault_pipe = Param('fault_pipe', fault_pipe_arr, dtype=int)
            m.leak_alg = Param('leak_alg', leak_alg_fault, dtype=int)
            m.qleak1 = Param('qleak1', qleak1_fault, dtype=int)
            m.qleak2 = Param('qleak2', qleak2_fault, dtype=int)
            m.leak_lm2 = Param('leak_lm2', leak_lm2, dtype=int)
            m.leak_lm1 = Param('leak_lm1', leak_lm1, dtype=int)
            m.leak_rp1 = Param('leak_rp1', leak_rp1, dtype=int)
            m.leak_rp2 = Param('leak_rp2', leak_rp2, dtype=int)

            f_idx = Idx('f_idx', n_fault)
            # leak_bd_left: replaces weno3-q_pipe_leak3_1.
            #   S*p[il] + va*qleak1 + S*p[il-2] + va*q[il-2]
            #   - 2*(S*p[il-1] + va*q[il-1]) = 0
            m.leak_bd_left = LoopEqn(
                'leak_bd_left', outer_index=f_idx,
                body=(m.Area[m.fault_pipe[f_idx]] * m.p_all[m.leak_alg[f_idx]]
                      + m.va * m.q_all[m.qleak1[f_idx]]
                      + m.Area[m.fault_pipe[f_idx]] * m.p_all[m.leak_lm2[f_idx]]
                      + m.va * m.q_all[m.leak_lm2[f_idx]]
                      - 2 * (m.Area[m.fault_pipe[f_idx]] * m.p_all[m.leak_lm1[f_idx]]
                             + m.va * m.q_all[m.leak_lm1[f_idx]])),
                model=m)
            # leak_bd_right: replaces weno3-q_pipe_leak3_2.
            #   S*p[il+2] - va*q[il+2] + S*p[il] - va*qleak2
            #   - 2*(S*p[il+1] - va*q[il+1]) = 0
            m.leak_bd_right = LoopEqn(
                'leak_bd_right', outer_index=f_idx,
                body=(m.Area[m.fault_pipe[f_idx]] * m.p_all[m.leak_rp2[f_idx]]
                      - m.va * m.q_all[m.leak_rp2[f_idx]]
                      + m.Area[m.fault_pipe[f_idx]] * m.p_all[m.leak_alg[f_idx]]
                      - m.va * m.q_all[m.qleak2[f_idx]]
                      - 2 * (m.Area[m.fault_pipe[f_idx]] * m.p_all[m.leak_rp1[f_idx]]
                             - m.va * m.q_all[m.leak_rp1[f_idx]])),
                model=m)

            # Per-fault scalar Eqns for the leak boundary itself.
            # Constants from leakage_ngs_pipe_weno3 (broken_pipe/weno3_broken.py).
            T0_leak = 293.0
            Mass_leak = 17.1e-3
            Z_leak = 1.0
            R_leak = 8.314
            Hcr_leak = 1.3
            C0_leak = 0.61
            Pa_leak = 101e3
            sonic_coef = (Mass_leak / (Z_leak * R_leak * T0_leak)
                          * Hcr_leak * (2 / (Hcr_leak + 1))
                          ** ((Hcr_leak + 1) / (Hcr_leak - 1))) ** 0.5
            if bundle_leak:
                # Bundled leak BC: ONE LoopEqn pair over all fault pipes
                # instead of a scalar Eqn pair per fault pipe. Collapses
                # the 2*n_fault generated ``inner_F``/``inner_J``
                # functions (``weno3_pipe{j}_leakage_bd1/bd2``) into two,
                # cutting both compile time and per-step F/J evaluation
                # cost. The per-pipe ``leak_rate``/``is_sonic`` scalars
                # become vector Params indexed over fault pipes. Arm the
                # leak from the case study by replacing the whole vector
                # param, writing the ramp into the faulted pipe's slot:
                #   slot = list(fault_pipe_index).index(faulted_pipe)
                #   mdl.p['leak_rate_all'] = TimeSeriesParam(
                #       'leak_rate_all', v_series=[0, 0, 1, 1],
                #       time_series=[0, t0, t0 + ramp, tend],
                #       index=[slot], value=np.zeros(n_fault))
                Ah_all = np.array(
                    [np.pi * (float(leak_diameter[fi]) / 2) ** 2
                     for fi in range(n_fault)])
                m.Ah_leak = Param('Ah_leak', Ah_all)
                m.is_sonic_leak = Param('is_sonic_leak', np.ones(n_fault))
                m.leak_rate_all = TimeSeriesParam(
                    'leak_rate_all',
                    v_series=[0.0, 0.0], time_series=[0.0, 3 * 3600],
                    index=np.array([0]), value=np.zeros(n_fault))
                P2 = m.p_all[m.leak_alg[f_idx]]
                m.weno3_leak_bd1 = LoopEqn(
                    'weno3_leak_bd1', outer_index=f_idx,
                    body=(m.q_all[m.leak_alg[f_idx]]
                          - m.leak_rate_all[f_idx] * (
                              m.is_sonic_leak[f_idx] * m.Ah_leak[f_idx]
                              * P2 * sonic_coef
                              + (1 - m.is_sonic_leak[f_idx]) * C0_leak
                                * m.Ah_leak[f_idx] * P2
                                * (2 * Mass_leak / (Z_leak * R_leak * T0_leak)
                                   * Hcr_leak / (Hcr_leak - 1)
                                   * ((Pa_leak / P2) ** (2 / Hcr_leak)
                                      - (Pa_leak / P2)
                                      ** ((Hcr_leak + 1) / Hcr_leak))) ** 0.5)),
                    model=m)
                m.weno3_leak_bd2 = LoopEqn(
                    'weno3_leak_bd2', outer_index=f_idx,
                    body=(m.q_all[m.qleak1[f_idx]]
                          - m.q_all[m.qleak2[f_idx]]
                          - m.q_all[m.leak_alg[f_idx]]),
                    model=m)
            else:
                for fi, j in enumerate(fault_pipe_arr):
                    pipe_name = f'pipe{int(j)}_leakage'
                    d_leak = float(leak_diameter[fi])
                    Ah_j = np.pi * (d_leak / 2) ** 2
                    is_sonic = Param(f'is_sonic_{pipe_name}', value=1)
                    m.__dict__[is_sonic.name] = is_sonic
                    leak_rate = TimeSeriesParam(
                        f'leak_rate_{pipe_name}',
                        v_series=[0, 0],
                        time_series=[0, 3 * 3600])
                    m.__dict__[leak_rate.name] = leak_rate

                    P2 = m.p_all[int(leak_alg_pipe[j])]
                    qjleak = leak_rate * (
                        is_sonic * Ah_j * P2 * sonic_coef
                        + (1 - is_sonic) * C0_leak * Ah_j * P2
                          * (2 * Mass_leak / (Z_leak * R_leak * T0_leak)
                             * Hcr_leak / (Hcr_leak - 1)
                             * ((Pa_leak / P2) ** (2 / Hcr_leak)
                                - (Pa_leak / P2) ** ((Hcr_leak + 1) / Hcr_leak))) ** 0.5)
                    m.__dict__[f'weno3_{pipe_name}_bd1'] = Eqn(
                        f'weno3_{pipe_name}_bd1',
                        m.q_all[int(leak_alg_pipe[j])] - qjleak)
                    m.__dict__[f'weno3_{pipe_name}_bd2'] = Eqn(
                        f'weno3_{pipe_name}_bd2',
                        m.q_all[int(qleak1_pipe[j])]
                        - m.q_all[int(qleak2_pipe[j])]
                        - m.q_all[int(leak_alg_pipe[j])])

        # --- cross-pipe LoopOde for PDE ---
        _add_gas_pipe_loopode(m, M, n_pipe, n_state_per, state_offsets,
                              is_fault, leak_cell,
                              bc_inlet_pos, bc_outlet_pos,
                              leak_alg_pipe, qleak1_pipe, qleak2_pipe,
                              total_state, va, dx)
        return m


def _add_gas_pipe_loopode(m, M, n_pipe, n_state_per, state_offsets,
                          is_fault, leak_cell,
                          bc_inlet_pos, bc_outlet_pos,
                          leak_alg_pipe, qleak1_pipe, qleak2_pipe,
                          total_state, va, dx):
    """Build ONE cross-pipe LoopOde each for q and p.

    Stencil
    -------
    * WENO3 with a five-point stencil for interior cells.
    * TVD1 (3-point) for boundary cells: the left BC neighbour
      (k = 0, original cell 1), the right BC neighbour
      (k = n_state-1, original cell M-1), and for fault pipes
      also the two cells immediately bracketing the leak
      (original cells ``il-1`` and ``il+1``).

    Cell numbering
    --------------
    Local state index ``k`` maps to original pipe cell ``c``:

    * No-fault pipe:  ``c = k + 1``        (cells 1..M-1).
    * Fault pipe:     ``c = k + 1`` for ``k < il-1`` (upstream
      segment, cells 1..il-1); ``c = k + 2`` for ``k >= il-1``
      (downstream segment, cells il+1..M-1).

    Neighbour-position Params
    -------------------------
    p and q stencils share ``pos_m2`` / ``pos_p2`` (always identical:
    the leak cell ``p[il]`` / ``q[il]`` is read directly by the WENO3
    stencils at cells ``il-2`` and ``il+2``). They diverge in
    ``pos_m1`` / ``pos_p1`` ONLY at the TVD1 cells immediately
    bracketing the leak, where the q-side reads ``qleak1`` (at
    ``c = il-1``) or ``qleak2`` (at ``c = il+1``) in place of the
    leak-cell q. Two pairs of pos-Params are therefore emitted:
    ``pos_p_m1`` / ``pos_p_p1`` (p stencil) and ``pos_q_m1`` /
    ``pos_q_p1`` (q stencil); for all non-fault cells the two pairs
    contain identical values.

    Single ``cell_is_tvd1`` mask selects TVD1 vs WENO3; pos_m2 and
    pos_p2 are set to ``g`` (dummy) at TVD1 cells, since the
    body's WENO3 term is multiplied out by ``(1 - mask)``.
    """
    from SolMuseum.pde.gas.weno3.weno_pipe import weno_odeq, weno_odep

    pos_p_m1 = np.empty(total_state, dtype=np.int64)
    pos_q_m1 = np.empty(total_state, dtype=np.int64)
    pos_p_p1 = np.empty(total_state, dtype=np.int64)
    pos_q_p1 = np.empty(total_state, dtype=np.int64)
    pos_m2 = np.empty(total_state, dtype=np.int64)
    pos_p2 = np.empty(total_state, dtype=np.int64)
    cell_pipe = np.empty(total_state, dtype=np.int64)
    cell_is_tvd1 = np.zeros(total_state)

    for j in range(n_pipe):
        Mj = int(M[j])
        soff = int(state_offsets[j])
        ns = int(n_state_per[j])
        bc_in = int(bc_inlet_pos[j])
        bc_out = int(bc_outlet_pos[j])
        if is_fault[j]:
            il = int(leak_cell[j])
            leak_alg = int(leak_alg_pipe[j])
            qleak1 = int(qleak1_pipe[j])
            qleak2 = int(qleak2_pipe[j])
        else:
            il = leak_alg = qleak1 = qleak2 = -1

        for k in range(ns):
            g = soff + k
            cell_pipe[g] = j
            # original cell number
            if is_fault[j] and k >= il - 1:
                c = k + 2
            else:
                c = k + 1

            is_left = (c == 1)
            is_right = (c == Mj - 1)
            is_leak_left = is_fault[j] and (c == il - 1)   # upstream right edge
            is_leak_right = is_fault[j] and (c == il + 1)  # downstream left edge
            tvd1 = is_left or is_right or is_leak_left or is_leak_right

            # pos_m1 (original cell c-1)
            if is_left:
                pos_p_m1[g] = bc_in
                pos_q_m1[g] = bc_in
            elif is_fault[j] and c - 1 == il:
                pos_p_m1[g] = leak_alg
                pos_q_m1[g] = qleak2
            else:
                pos_p_m1[g] = g - 1
                pos_q_m1[g] = g - 1

            # pos_p1 (original cell c+1)
            if is_right:
                pos_p_p1[g] = bc_out
                pos_q_p1[g] = bc_out
            elif is_fault[j] and c + 1 == il:
                pos_p_p1[g] = leak_alg
                pos_q_p1[g] = qleak1
            else:
                pos_p_p1[g] = g + 1
                pos_q_p1[g] = g + 1

            # pos_m2 / pos_p2 only matter for WENO3 cells.
            if tvd1:
                pos_m2[g] = g
                pos_p2[g] = g
            else:
                # pos_m2 (original cell c-2)
                if c == 2:
                    pos_m2[g] = bc_in
                elif is_fault[j] and c - 2 == il:
                    pos_m2[g] = leak_alg
                else:
                    pos_m2[g] = g - 2
                # pos_p2 (original cell c+2)
                if c == Mj - 2:
                    pos_p2[g] = bc_out
                elif is_fault[j] and c + 2 == il:
                    pos_p2[g] = leak_alg
                else:
                    pos_p2[g] = g + 2

            if tvd1:
                cell_is_tvd1[g] = 1.0

    m.cell_pipe_g = Param('cell_pipe_g', cell_pipe, dtype=int)
    m.cell_is_tvd1 = Param('cell_is_tvd1', cell_is_tvd1)
    m.pos_p_m1 = Param('pos_p_m1', pos_p_m1, dtype=int)
    m.pos_q_m1 = Param('pos_q_m1', pos_q_m1, dtype=int)
    m.pos_p_p1 = Param('pos_p_p1', pos_p_p1, dtype=int)
    m.pos_q_p1 = Param('pos_q_p1', pos_q_p1, dtype=int)
    m.pos_m2 = Param('pos_m2', pos_m2, dtype=int)
    m.pos_p2 = Param('pos_p2', pos_p2, dtype=int)

    g = Idx('g_gas', total_state)
    p_ib = sp.IndexedBase('p_all')
    q_ib = sp.IndexedBase('q_all')
    j_idx = m.cell_pipe_g[g]
    mask_bdy = m.cell_is_tvd1[g]

    # TVD1 (boundary cells)
    tvd_q = (
        -m.Area[j_idx] * (p_ib[m.pos_p_p1[g]] - p_ib[m.pos_p_m1[g]]) / (2 * dx)
        - m.lam_gas_pipe[j_idx] * m.va ** 2 * q_ib[g] * Abs(q_ib[g])
        / (2 * m.D[j_idx] * m.Area[j_idx] * p_ib[g])
    )
    tvd_p = -m.va ** 2 / m.Area[j_idx] * (q_ib[m.pos_q_p1[g]] - q_ib[m.pos_q_m1[g]]) / (2 * dx)

    # WENO3 interior cells
    weno_q = weno_odeq(
        p_ib[m.pos_m2[g]], p_ib[m.pos_p_m1[g]], p_ib[g],
        p_ib[m.pos_p_p1[g]], p_ib[m.pos_p2[g]],
        q_ib[m.pos_m2[g]], q_ib[m.pos_q_m1[g]], q_ib[g],
        q_ib[m.pos_q_p1[g]], q_ib[m.pos_p2[g]],
        m.Area[j_idx], m.va, m.lam_gas_pipe[j_idx], m.D[j_idx], dx)
    weno_p = weno_odep(
        p_ib[m.pos_m2[g]], p_ib[m.pos_p_m1[g]], p_ib[g],
        p_ib[m.pos_p_p1[g]], p_ib[m.pos_p2[g]],
        q_ib[m.pos_m2[g]], q_ib[m.pos_q_m1[g]], q_ib[g],
        q_ib[m.pos_q_p1[g]], q_ib[m.pos_p2[g]],
        m.Area[j_idx], m.va, m.lam_gas_pipe[j_idx], m.D[j_idx], dx)

    m.gas_pipe_q = LoopOde(
        'gas_pipe_q', outer_index=g,
        body=mask_bdy * tvd_q + (1 - mask_bdy) * weno_q,
        diff_var=m.q_all[0:total_state], model=m)
    m.gas_pipe_p = LoopOde(
        'gas_pipe_p', outer_index=g,
        body=mask_bdy * tvd_p + (1 - mask_bdy) * weno_p,
        diff_var=m.p_all[0:total_state], model=m)
