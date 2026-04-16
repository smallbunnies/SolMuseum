import sympy as sp
import numpy as np
from scipy.sparse import csc_array
from Solverz import Eqn, Param, Model, Idx, LoopEqn, LoopOde, Sum
from Solverz import Var, Abs
from Solverz.utilities.type_checker import is_number
from SolUtil import GasFlow

from SolMuseum.pde import ngs_pipe, leakage_pipe, rupture_pipe


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
            loopeqn=True):
        if loopeqn and fault_type is None:
            return self._mdl_loopeqn(dx, dt, method)
        return self._mdl_legacy(dx, dt, method, fault_type,
                                fault_pipe_index, fault_loc_index,
                                leak_diameter)

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
    def _mdl_loopeqn(self, dx, dt, method):
        """Build the gas-network model using LoopEqn / LoopOde.

        Flat layout
        -----------
        ``p_all`` / ``q_all`` pack all pipes into one Var each:

            [ state_cells_pipe0 | ... | bc_inlet_cells | bc_outlet_cells ]

        State cells (cells 1..M-1, i.e. M-1 per pipe) are contiguous
        at the front; BC inlet cells (cell 0) and BC outlet cells
        (cell M) sit at the tail.

        Equation structure
        ------------------
        * **mass_continuity** — LoopEqn over all nodes
        * **pressure_inlet / pressure_outlet** — LoopEqn over pipes
        * **slack_pressure** — scalar Eqn per slack node
        * **bd_right / bd_left** — LoopEqn over pipes (characteristic BC)
        * **gas_pipe_q / gas_pipe_p** — cross-pipe LoopOde (weno3
          interior + TVD1 boundary via mask)
        """
        gf = self.gf
        n_node = gf.n_node
        n_pipe = gf.n_pipe
        va = gf.va
        L = gf.L
        M = np.floor(L / dx).astype(int)

        m = Model()
        m.Pi = Var('Pi', value=gf.Pi * 1e6)
        m.fs = Var('fs', value=gf.fs)
        m.fl = Var('fl', value=gf.fl)
        m.D = Param('D', value=gf.D)
        m.Area = Param('Area', np.pi * (gf.D / 2) ** 2)
        m.lam_gas_pipe = Param('lam_gas_pipe', value=gf.lam)
        m.va = Param('va', va)

        # --- flat layout: state cells first, BC cells at end ---
        # State cells per pipe: cells 1..M-1 (M-1 cells)
        # BC inlet: cell 0; BC outlet: cell M
        state_offsets = np.zeros(n_pipe + 1, dtype=int)
        for j in range(n_pipe):
            state_offsets[j + 1] = state_offsets[j] + int(M[j]) - 1
        total_state = int(state_offsets[-1])
        total_len = total_state + 2 * n_pipe  # +inlet +outlet

        p0_all = np.empty(total_len)
        q0_all = np.empty(total_len)
        for j in range(n_pipe):
            Mj = int(M[j])
            soff = int(state_offsets[j])
            p_profile = np.linspace(
                gf.Pi[gf.rpipe_from[j]] * 1e6,
                gf.Pi[gf.rpipe_to[j]] * 1e6, Mj + 1)
            p0_all[soff:soff + Mj - 1] = p_profile[1:Mj]
            p0_all[total_state + j] = p_profile[0]           # BC inlet
            p0_all[total_state + n_pipe + j] = p_profile[Mj]  # BC outlet
            q0_all[soff:soff + Mj - 1] = gf.f[j]
            q0_all[total_state + j] = gf.f[j]
            q0_all[total_state + n_pipe + j] = gf.f[j]
        m.p_all = Var('p_all', p0_all)
        m.q_all = Var('q_all', q0_all)

        # --- pipe-level index Params ---
        bc_inlet_pos = (total_state + np.arange(n_pipe)).astype(np.int64)
        bc_outlet_pos = (total_state + n_pipe + np.arange(n_pipe)).astype(np.int64)
        # last state cell of pipe j = state_offsets[j] + M[j] - 2
        last_state_pos = (state_offsets[:-1] + M - 2).astype(np.int64)
        first_state_pos = state_offsets[:-1].astype(np.int64)
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

        # --- sparse node×q_all incidence for mass continuity ---
        # Q_net_gas[node, k] = +1 at k=bc_inlet_pos[j] for outgoing
        #   pipe j, −1 at k=bc_outlet_pos[j] for incoming pipe j.
        # Using this directly in Sum(Q_net_gas[i, k] * q_all[k], k)
        # gives a standard Mat_Mul pattern whose J sparsity the
        # analyzer handles natively (no double-indirection δ).
        from scipy.sparse import lil_array as _lil
        Q_net = _lil((n_node, total_len))
        for node in gf.gc['G'].nodes:
            for edge in gf.gc['G'].in_edges(node, data=True):
                if edge[2]['type'] == 1:
                    Q_net[node, bc_outlet_pos[edge[2]['idx']]] = -1.0
            for edge in gf.gc['G'].out_edges(node, data=True):
                if edge[2]['type'] == 1:
                    Q_net[node, bc_inlet_pos[edge[2]['idx']]] = 1.0
        m.Q_net_gas = Param('Q_net_gas', csc_array(Q_net), dim=2, sparse=True)

        # --- shared LoopEqn indices ---
        i_n = Idx('i_n', n_node)
        p_p = Idx('p_p', n_pipe)
        k_q = Idx('k_q', total_len)

        # --- mass continuity LoopEqn ---
        m.mass_continuity = LoopEqn(
            'mass_continuity', outer_index=i_n,
            body=(-m.fs[i_n] + m.fl[i_n]
                  + Sum(m.Q_net_gas[i_n, k_q] * m.q_all[k_q], k_q)),
            model=m)

        # --- pressure coupling LoopEqns ---
        m.pressure_inlet = LoopEqn(
            'pressure_inlet', outer_index=p_p,
            body=m.Pi[m.gas_pipe_from[p_p]] - m.p_all[m.bc_inlet_pos[p_p]],
            model=m)
        m.pressure_outlet = LoopEqn(
            'pressure_outlet', outer_index=p_p,
            body=m.Pi[m.gas_pipe_to[p_p]] - m.p_all[m.bc_outlet_pos[p_p]],
            model=m)

        # --- slack pressure ---
        for i in gf.slack:
            m.__dict__[f'node_pressure_{i}'] = Eqn(
                f'node_pressure_{i}',
                m.Pi[i] - gf.Piset[gf.slack.tolist().index(i)] * 1e6)

        # --- characteristic BC LoopEqns (bd1, bd2) ---
        # bd_right: S*p[M] + va*q[M] + S*p[M-2] + va*q[M-2]
        #           - 2*(S*p[M-1] + va*q[M-1]) = 0
        # In flat layout: p[M]=bc_outlet, p[M-2]=last_state-1,
        #                 p[M-1]=last_state
        # Pre-computed neighbour positions for the characteristic BCs
        # so the LoopEqn body avoids Indexed±const arithmetic that
        # the sparsity analyzer can't classify.
        m.last_state = Param('last_state', last_state_pos, dtype=int)
        m.second_last_state = Param('second_last_state',
                                     (last_state_pos - 1).astype(np.int64), dtype=int)
        m.first_state = Param('first_state', first_state_pos, dtype=int)
        m.second_state = Param('second_state',
                                (first_state_pos + 1).astype(np.int64), dtype=int)

        # bd_right: S*p[M] + va*q[M] + S*p[M-2] + va*q[M-2]
        #           - 2*(S*p[M-1] + va*q[M-1]) = 0
        m.bd_right = LoopEqn(
            'bd_right', outer_index=p_p,
            body=(m.Area[p_p] * m.p_all[m.bc_outlet_pos[p_p]]
                  + m.va * m.q_all[m.bc_outlet_pos[p_p]]
                  + m.Area[p_p] * m.p_all[m.second_last_state[p_p]]
                  + m.va * m.q_all[m.second_last_state[p_p]]
                  - 2 * (m.Area[p_p] * m.p_all[m.last_state[p_p]]
                         + m.va * m.q_all[m.last_state[p_p]])),
            model=m)
        # bd_left: S*p[2] - va*q[2] + S*p[0] - va*q[0]
        #          - 2*(S*p[1] - va*q[1]) = 0
        m.bd_left = LoopEqn(
            'bd_left', outer_index=p_p,
            body=(m.Area[p_p] * m.p_all[m.second_state[p_p]]
                  - m.va * m.q_all[m.second_state[p_p]]
                  + m.Area[p_p] * m.p_all[m.bc_inlet_pos[p_p]]
                  - m.va * m.q_all[m.bc_inlet_pos[p_p]]
                  - 2 * (m.Area[p_p] * m.p_all[m.first_state[p_p]]
                         - m.va * m.q_all[m.first_state[p_p]])),
            model=m)

        # --- cross-pipe LoopOde for PDE ---
        _add_gas_pipe_loopode(m, M, n_pipe, state_offsets,
                               total_state, va, dx)
        return m


def _add_gas_pipe_loopode(m, M, n_pipe, state_offsets, total_state, va, dx):
    """Build ONE cross-pipe LoopOde each for q and p.

    The weno3 5-point stencil covers interior cells (local k ∈ [1, M-3]);
    boundary cells (k=0 and k=M-2) use the TVD first-order 3-point
    stencil. Pre-computed Param arrays resolve stencil neighbour
    positions and boundary masks.

    State cell numbering
    --------------------
    Local state index ``k`` maps to original pipe cell ``k + 1``
    (cells 1..M-1; cells 0 and M are BC-pinned).

    * ``pos_m1[g]`` — original cell k, i.e. one left of current.
      At k=0 this is the BC inlet cell.
    * ``pos_m2[g]`` — original cell k-1, two left.
      At k=0 clamped to g (dummy, masked by is_left).
      At k=1 this is the BC inlet cell.
    * ``pos_p1[g]`` — original cell k+2, one right.
      At k=M-2 this is the BC outlet cell.
    * ``pos_p2[g]`` — original cell k+3, two right.
      At k=M-2 clamped to g (dummy, masked by is_right).
      At k=M-3 this is the BC outlet cell.
    """
    from SolMuseum.pde.gas.weno3.weno_pipe import weno_odeq, weno_odep

    pos_m1 = np.empty(total_state, dtype=np.int64)
    pos_m2 = np.empty(total_state, dtype=np.int64)
    pos_p1 = np.empty(total_state, dtype=np.int64)
    pos_p2 = np.empty(total_state, dtype=np.int64)
    cell_pipe = np.empty(total_state, dtype=np.int64)
    cell_is_left = np.zeros(total_state)
    cell_is_right = np.zeros(total_state)

    for j in range(n_pipe):
        bc_in = total_state + j
        bc_out = total_state + n_pipe + j
        soff = int(state_offsets[j])
        n_state = int(M[j]) - 1  # cells 1..M-1

        for k in range(n_state):
            g = soff + k
            cell_pipe[g] = j
            # pos_m1: original cell k (one left of current cell k+1)
            pos_m1[g] = bc_in if k == 0 else g - 1
            # pos_m2: original cell k-1
            if k == 0:
                pos_m2[g] = g       # dummy (masked by is_left)
            elif k == 1:
                pos_m2[g] = bc_in   # cell 0 = BC inlet
            else:
                pos_m2[g] = g - 2
            # pos_p1: original cell k+2 (one right of current cell k+1)
            pos_p1[g] = bc_out if k == n_state - 1 else g + 1
            # pos_p2: original cell k+3
            if k == n_state - 1:
                pos_p2[g] = g       # dummy (masked by is_right)
            elif k == n_state - 2:
                pos_p2[g] = bc_out  # cell M = BC outlet
            else:
                pos_p2[g] = g + 2
            if k == 0:
                cell_is_left[g] = 1.0
            if k == n_state - 1:
                cell_is_right[g] = 1.0

    m.cell_pipe_g = Param('cell_pipe_g', cell_pipe, dtype=int)
    m.cell_is_left = Param('cell_is_left', cell_is_left)
    m.cell_is_right = Param('cell_is_right', cell_is_right)
    m.pos_m1 = Param('pos_m1', pos_m1, dtype=int)
    m.pos_m2 = Param('pos_m2', pos_m2, dtype=int)
    m.pos_p1 = Param('pos_p1', pos_p1, dtype=int)
    m.pos_p2 = Param('pos_p2', pos_p2, dtype=int)

    g = Idx('g_gas', total_state)
    p_ib = sp.IndexedBase('p_all')
    q_ib = sp.IndexedBase('q_all')
    j_idx = m.cell_pipe_g[g]
    mask_bdy = m.cell_is_left[g] + m.cell_is_right[g]

    # TVD first-order 3-point stencil (boundary cells):
    #   dq/dt = -S*(pp1-pm1)/(2*dx) - lam*va^2*q0*|q0|/(2*D*S*p0)
    #   dp/dt = -va^2/S * (qp1-qm1)/(2*dx)
    tvd_q = (
        -m.Area[j_idx] * (p_ib[m.pos_p1[g]] - p_ib[m.pos_m1[g]]) / (2 * dx)
        - m.lam_gas_pipe[j_idx] * m.va ** 2 * q_ib[g] * Abs(q_ib[g])
        / (2 * m.D[j_idx] * m.Area[j_idx] * p_ib[g])
    )
    tvd_p = -m.va ** 2 / m.Area[j_idx] * (q_ib[m.pos_p1[g]] - q_ib[m.pos_m1[g]]) / (2 * dx)

    # weno3 5-point stencil (interior cells):
    weno_q = weno_odeq(
        p_ib[m.pos_m2[g]], p_ib[m.pos_m1[g]], p_ib[g],
        p_ib[m.pos_p1[g]], p_ib[m.pos_p2[g]],
        q_ib[m.pos_m2[g]], q_ib[m.pos_m1[g]], q_ib[g],
        q_ib[m.pos_p1[g]], q_ib[m.pos_p2[g]],
        m.Area[j_idx], m.va, m.lam_gas_pipe[j_idx], m.D[j_idx], dx)
    weno_p = weno_odep(
        p_ib[m.pos_m2[g]], p_ib[m.pos_m1[g]], p_ib[g],
        p_ib[m.pos_p1[g]], p_ib[m.pos_p2[g]],
        q_ib[m.pos_m2[g]], q_ib[m.pos_m1[g]], q_ib[g],
        q_ib[m.pos_p1[g]], q_ib[m.pos_p2[g]],
        m.Area[j_idx], m.va, m.lam_gas_pipe[j_idx], m.D[j_idx], dx)

    m.gas_pipe_q = LoopOde(
        'gas_pipe_q', outer_index=g,
        body=mask_bdy * tvd_q + (1 - mask_bdy) * weno_q,
        diff_var=m.q_all[0:total_state], model=m)
    m.gas_pipe_p = LoopOde(
        'gas_pipe_p', outer_index=g,
        body=mask_bdy * tvd_p + (1 - mask_bdy) * weno_p,
        diff_var=m.p_all[0:total_state], model=m)
