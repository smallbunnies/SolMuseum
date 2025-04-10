import numpy as np
from numba import njit
from ..minmod_limiter import *

@njit(cache=True)
def kt2_ode(Tm2, Tm1, T0, Tp1, m, lam, rho, Cp, S, Tamb, theta, dx):
    return m*(-1/2*dx*minmod(theta*(T0 - Tm1)/dx, (1/2)*(-Tm1 + Tp1)/dx, theta*(-T0 + Tp1)/dx) + (1/2)*dx*minmod(theta*(Tm1 - Tm2)/dx, (1/2)*(T0 - Tm2)/dx, theta*(T0 - Tm1)/dx) - T0 + Tm1)/(S*dx*rho) - lam*(-Tamb + T0)/(Cp*S*rho)

@njit(cache=True)
def dkt2_odedT0(Tm2, Tm1, T0, Tp1, m, lam, rho, Cp, S, Tamb, theta, dx):
    return m*((1/2)*dx*switch_minmod(0, (1/2)/dx, theta/dx, minmod_flag(theta*(Tm1 - Tm2)/dx, (1/2)*(T0 - Tm2)/dx, theta*(T0 - Tm1)/dx)) - 1/2*dx*switch_minmod(theta/dx, 0, -theta/dx, minmod_flag(theta*(T0 - Tm1)/dx, (1/2)*(-Tm1 + Tp1)/dx, theta*(-T0 + Tp1)/dx)) - 1)/(S*dx*rho) - lam/(Cp*S*rho)

@njit(cache=True)
def dkt2_odedTm1(Tm2, Tm1, T0, Tp1, m, lam, rho, Cp, S, Tamb, theta, dx):
    return m*(-1/2*dx*switch_minmod(-theta/dx, -(1/2)/dx, 0, minmod_flag(theta*(T0 - Tm1)/dx, (1/2)*(-Tm1 + Tp1)/dx, theta*(-T0 + Tp1)/dx)) + (1/2)*dx*switch_minmod(theta/dx, 0, -theta/dx, minmod_flag(theta*(Tm1 - Tm2)/dx, (1/2)*(T0 - Tm2)/dx, theta*(T0 - Tm1)/dx)) + 1)/(S*dx*rho)

@njit(cache=True)
def dkt2_odedTp1(Tm2, Tm1, T0, Tp1, m, lam, rho, Cp, S, Tamb, theta, dx):
    return -1/2*m*switch_minmod(0, (1/2)/dx, theta/dx, minmod_flag(theta*(T0 - Tm1)/dx, (1/2)*(-Tm1 + Tp1)/dx, theta*(-T0 + Tp1)/dx))/(S*rho)

@njit(cache=True)
def dkt2_odedTm2(Tm2, Tm1, T0, Tp1, m, lam, rho, Cp, S, Tamb, theta, dx):
    return (1/2)*m*switch_minmod(-theta/dx, -(1/2)/dx, 0, minmod_flag(theta*(Tm1 - Tm2)/dx, (1/2)*(T0 - Tm2)/dx, theta*(T0 - Tm1)/dx))/(S*rho)