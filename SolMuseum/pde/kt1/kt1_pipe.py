from SolMuseum.pde.util import *
from Solverz.sym_algebra.functions import MulVarFunc


class kt1_ode0(MulVarFunc):
    """
    pm1, p0, pp1, qm1, q0, qp1, S, va, lam, D, dx
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return dkt1_ode0dpm1(*self.args)
        elif argindex == 2:
            return dkt1_ode0dp0(*self.args)
        elif argindex == 3:
            return dkt1_ode0dpp1(*self.args)
        elif argindex == 4:
            return dkt1_ode0dqm1(*self.args)
        elif argindex == 5:
            return dkt1_ode0dq0(*self.args)
        elif argindex == 6:
            return dkt1_ode0dqp1(*self.args)
        else:
            return Integer(0)


class dkt1_ode0dpm1(MulVarFunc):
    pass


class dkt1_ode0dp0(MulVarFunc):
    pass


class dkt1_ode0dpp1(MulVarFunc):
    pass


class dkt1_ode0dqm1(MulVarFunc):
    pass


class dkt1_ode0dq0(MulVarFunc):
    pass


class dkt1_ode0dqp1(MulVarFunc):
    pass


class kt1_ode1(MulVarFunc):
    """
    pm1, p0, pp1, qm1, q0, qp1, S, va, dx
    """
    pass


class kt1_ae0(MulVarFunc):
    """
    pm1, p0, pp1, qm1, q0, qp1, S, va, dx
    """
    pass


class kt1_ae1(MulVarFunc):
    """
    pm1, p0, pp1, qm1, q0, qp1, S, va, d
    """
    pass
