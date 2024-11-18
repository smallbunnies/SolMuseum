from SolMuseum.pde.gas.util import *
from SolMuseum.pde.basic import SolPde

variables = ['pm2', 'pm1', 'p0', 'pp1', 'pp2', 'qm2', 'qm1', 'q0', 'qp1', 'qp2']
dweno_odepd = [type(f'dweno_odepd{varname}', (SolPde,), {}) for varname in variables]
dweno_odeqd = [type(f'dweno_odeqd{varname}', (SolPde,), {}) for varname in variables]

for i, varname in enumerate(variables):
    setattr(dweno_odepd[i], 'arglength', 15)
    setattr(dweno_odeqd[i], 'arglength', 15)


class weno_odep(SolPde):
    """
    pm2, pm1, p0, pp1, pp2, qm2, qm1, q0, qp1, qp2, S, va, lam, D, dx
    """

    arglength = 15

    def fdiff(self, argindex=1):
        if argindex in range(1, len(variables) + 1):
            return dweno_odepd[argindex - 1](*self.args)
        else:
            return Integer(0)


class weno_odeq(SolPde):
    """
    pm2, pm1, p0, pp1, pp2, qm2, qm1, q0, qp1, qp2, S, va, lam, D, dx
    """

    arglength = 15

    def fdiff(self, argindex=1):
        if argindex in range(1, len(variables) + 1):
            return dweno_odeqd[argindex - 1](*self.args)
        else:
            return Integer(0)
