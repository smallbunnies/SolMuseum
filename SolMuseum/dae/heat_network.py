import numpy as np
from Solverz import Eqn, Param, Model
from Solverz import Var, Abs, heaviside, exp, Sign
from Solverz.utilities.type_checker import is_number
from SolUtil import DhsFlow

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
            Tend = (Tstart-Tamb[0])*attenuation+Tamb[0]
            Tsp0 = np.linspace(Tstart,
                               Tend,
                               M[j] + 1)

            # return pipe
            Tstart = self.df.Tr[self.df.pipe_to[j]]
            Tend = (Tstart-Tamb[0])*attenuation+Tamb[0]
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
        if len(self.df.pinloop) > 0:
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
                            's'+str(pipe),
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
                            'r'+str(pipe),
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
