from SolMuseum.pde.gas.util import *
from SolMuseum.pde.basic import SolPde

variables = ['Tm1', 'T0']
dkt1_oded = [type(f'dkt1_oded{varname}', (SolPde,), {}) for varname in variables]

for i, varname in enumerate(variables):
    setattr(dkt1_oded[i], 'arglength', 9)


class kt1_ode(SolPde):
    """
    Tm1, T0, m, lam, rho, Cp, S, Tamb, dx
    """

    arglength = 9

    def fdiff(self, argindex=1):
        if argindex in range(1, len(variables) + 1):
            return dkt1_oded[argindex - 1](*self.args)
        else:
            return Integer(0)
