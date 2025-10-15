import numpy as np
from Solverz import Eqn, Param, Model, TimeSeriesParam, Var, Abs, heaviside, exp, Sign
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
            dynamic_slack=False):
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

        # mass flow continuity
        for node in range(self.df.n_node):
            rhs = - m.min[node]
            for edge in self.df.G.in_edges(node, data=True):
                pipe = edge[2]['idx']
                rhs = rhs + m.m[pipe]
                idx = str(pipe)
                Trpj = m.__dict__['Trp_' + idx]
                m.__dict__[f'Return_pipe_inlet_temp_{pipe}'] = Eqn(f'Return_pipe_inlet_temp_{pipe}',
                                                                   Trpj[0] - m.Tr[node])

            for edge in self.df.G.out_edges(node, data=True):
                pipe = edge[2]['idx']
                rhs = rhs - m.m[pipe]
                idx = str(pipe)
                Tspj = m.__dict__['Tsp_' + idx]
                m.__dict__[f'Supply_pipe_inlet_temp_{pipe}'] = Eqn(f'Supply_pipe_inlet_temp_{pipe}',
                                                                   Tspj[0] - m.Ts[node])
            m.__dict__[f"Mass_flow_continuity_{node}"] = Eqn(f"Mass_flow_continuity_{node}", rhs)

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
