import numpy as np
from Solverz import Eqn, Param, Model
from Solverz import Var, Abs
from Solverz.utilities.type_checker import is_number
from SolUtil import DhsFlow

from ..pde import ngs_pipe
from warnings import warn
from ..util import rename_mdl


class gas_network:

    def __init__(self, df: DhsFlow):
        self.df = df
        self.df.run()

    def mdl(self,
            dx,
            method='weno3'):
        m = Model()
        m.Pi = Var('Pi', value=self.df.Pi * 1e6)  # node pressure
        m.fs = Var('fs', value=self.df.fs)
        m.fl = Var('fl', value=self.df.fl)
        va = self.df.va
        m.D = Param('D', value=self.df.D)
        m.Area = Param('Area', np.pi * (self.df.D / 2) ** 2)
        m.lam = Param('lam', value=self.df.lam)
        L = self.df.L
        dx = dx
        M = np.floor(L / dx).astype(int)
        for j in range(self.df.n_pipe):
            p0 = np.linspace(self.df.Pi[self.df.rpipe_from[j]]*1e6,
                             self.df.Pi[self.df.rpipe_to[j]]*1e6,
                             M[j] + 1)
            m.__dict__['p' + str(j)] = Var('p' + str(j), value=p0)
            m.__dict__['q' + str(j)] = Var('q' + str(j), value=self.df.f[j] * np.ones(M[j] + 1))

        # % method of lines
        for node in self.df.gc['G'].nodes:
            eqn_q = - m.fs[node] + m.fl[node]
            for edge in self.df.gc['G'].in_edges(node, data=True):
                pipe = edge[2]['idx']
                idx = str(pipe)
                ptype = edge[2]['type']
                qi = m.__dict__['q' + idx]
                pi = m.__dict__['p' + idx]
                if ptype == 1:
                    eqn_q = eqn_q - qi[M[pipe]]
                    m.__dict__[f'pressure_outlet_pipe{idx}'] = Eqn(f'Pressure node {node} pipe {idx} outlet',
                                                                   m.Pi[node] - pi[M[pipe]])

            for edge in self.df.gc['G'].out_edges(node, data=True):
                pipe = edge[2]['idx']
                idx = str(pipe)
                ptype = edge[2]['type']
                qi = m.__dict__['q' + idx]
                pi = m.__dict__['p' + idx]
                if ptype == 1:
                    eqn_q = eqn_q + qi[0]
                    m.__dict__[f'pressure_inlet_pipe{idx}'] = Eqn(f'Pressure node {node} pipe {idx} inlet',
                                                                  m.Pi[node] - pi[0])

            m.__dict__[f'mass_continuity_node{node}'] = Eqn('mass flow continuity of node {}'.format(node),
                                                            eqn_q)

        for i in self.df.slack:
            m.__dict__[f'node_pressure_{i}'] = Eqn(f'node_pressure_{i}',
                                                   m.Pi[i] - self.df.Pi_slack[i]*1e6)

        # difference and initialize variables
        for edge in self.df.gc['G'].edges(data=True):
            j = edge[2]['idx']
            Mj = M[j]
            pj = m.__dict__['p' + str(j)]
            qj = m.__dict__['q' + str(j)]
            Dj = m.D[j]
            Sj = m.Area[j]
            lamj = m.lam[j]

            ngs_p = ngs_pipe(pj,
                             qj,
                             lamj,
                             va,
                             Dj,
                             Sj,
                             dx,
                             0,
                             Mj,
                             f'pipe{j}',
                             method=method)

            for key, value in ngs_p.items():
                m.__dict__[key] = value

        return m
