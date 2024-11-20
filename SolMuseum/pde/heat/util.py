from Solverz import Eqn, Ode, AliasVar, TimeSeriesParam, Param
from Solverz import iVar, idx, Var, Abs
from Solverz.utilities.type_checker import is_integer, is_number
from ..basic import minmod


def ux(theta, um1, u, up1, dx):
    return minmod(theta * (u - um1) / dx, (up1 - um1) / (2 * dx), theta * (up1 - u) / dx)


def mol_kt2_rhs(T,
                m,
                lam,
                rho,
                Cp,
                S,
                Tamb,
                dx,
                start,
                end,
                order=2):
    """
    Four point stencil
    :param order:
    :param T:
    :param m:
    :param lam:
    :param rho:
    :param Cp:
    :param S:
    :param Tamb:
    :param dx:
    :param start:
    :param end:
    :return:
    """
    id0 = start
    id1 = end
    if id0 >= id1:
        raise ValueError('id0 should be smaller than id1!')

    if end - start == 1:
        Tm2, Tm1, T0, Tp1 = [T[id0 - 2], T[id0 - 1], T[id0], T[id0 + 1]]
    else:
        Tm2, Tm1, T0, Tp1 = [T[id0 - 2:id1 - 2], T[id0 - 1:id1 - 1], T[id0:id1], T[id0 + 1:id1 + 1]]

    theta = Param('theta', 1)
    if order == 2:
        Txj = ux(theta, Tm1, T0, Tp1, dx)
        Txjm1 = ux(theta, Tm2, Tm1, T0, dx)
    elif order == 1:
        Txj = 0
        Txjm1 = 0
    else:
        raise ValueError(f"Order {order} not supported!")

    rhs = m / (S * rho * dx) * (Tm1 + dx / 2 * Txjm1 - T0 - dx / 2 * Txj) - lam / (S * rho * Cp) * (T0 - Tamb)

    return rhs
