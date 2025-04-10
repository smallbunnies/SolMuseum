from SolMuseum.pde.gas.util import *
from SolMuseum.pde.basic import SolPde

variables = ['Tm2', 'Tm1', 'T0', 'Tp1']
dkt2_oded = [type(f'dkt2_oded{varname}', (SolPde,), {}) for varname in variables]

for i, varname in enumerate(variables):
    setattr(dkt2_oded[i], 'arglength', 12)


class kt2_ode(SolPde):
    """
    Tm2, Tm1, T0, Tp1, m, lam, rho, Cp, S, Tamb, theta, dx
    """

    arglength = 12

    def fdiff(self, argindex=1):
        if argindex in range(1, len(variables) + 1):
            return dkt2_oded[argindex - 1](*self.args)
        else:
            return Integer(0)
