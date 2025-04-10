import numpy as np
from numba import njit

@njit(cache=True)
def kt1_ode(Tm1, T0, m, lam, rho, Cp, S, Tamb, dx):
    return m*(-T0 + Tm1)/(S*dx*rho) - lam*(-Tamb + T0)/(Cp*S*rho)

@njit(cache=True)
def dkt1_odedT0(Tm1, T0, m, lam, rho, Cp, S, Tamb, dx):
    return -m/(S*dx*rho) - lam/(Cp*S*rho)

@njit(cache=True)
def dkt1_odedTm1(Tm1, T0, m, lam, rho, Cp, S, Tamb, dx):
    return m/(S*dx*rho)