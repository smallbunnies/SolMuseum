from SolMuseum.pde.gas.util import *
from .euler_broken import rupture_ngs_pipe_Euler, leakage_ngs_pipe_Euler
from .cdm_broken import rupture_ngs_pipe_cdm, leakage_ngs_pipe_cdm
from .kt1_broken import rupture_ngs_pipe_kt1, leakage_ngs_pipe_kt1
from .kt2_broken import rupture_ngs_pipe_kt2, leakage_ngs_pipe_kt2
from .cha_broken import rupture_ngs_pipe_cha, leakage_ngs_pipe_cha
from .weno3_broken import rupture_ngs_pipe_weno3, leakage_ngs_pipe_weno3


def rupture_pipe(p: Var,
                 q: Var,
                 lam,
                 va,
                 D,
                 S,
                 dx,
                 dt,
                 M,
                 pipe_name: str,
                 idx_leak,
                 method='weno3'):
    r"""
    The rupture fault model of ngs pipes, with the boundary condition

        .. math::
            p_i = p_a

    where $p_i$ is the pressure of the rupture location, $p_a$ is the atmospheric pressure.
    The fault model is based on the normal ngs pipe defined above.

    Parameters
    ==========

    idx_leak: Number

        The index of rupture location.

    method : str

        'euler' - Default, the euler scheme

        'cdm' - The central difference scheme

        'cha' - The method of characteristics

        'kt1' - The first order Kurganov-Tadmor semi-discretization scheme

        'kt2' - The second order Kurganov-Tadmor semi-discretization scheme

        'weno3' - The third order WENO semi-discretization scheme

    Returns
    =======

    artifact : dict()
        The dict of derived equations and variables.

    """

    if not is_integer(M):
        raise TypeError(f'M is {type(M)} instead of integer')

    if not is_number(dx):
        raise TypeError(f'dx is {type(dx)} instead of number')

    if not is_number(idx_leak):
        raise TypeError(f'idx_leak is {type(idx_leak)} instead of number')

    artifact = dict()

    match method:
        case 'euler':
            rupture_ngs_pipe_Euler(p,
                                   q,
                                   lam,
                                   va,
                                   D,
                                   S,
                                   dx,
                                   dt,
                                   M,
                                   pipe_name,
                                   idx_leak)

        case 'cdm':
            rupture_ngs_pipe_cdm(p,
                                 q,
                                 lam,
                                 va,
                                 D,
                                 S,
                                 dx,
                                 dt,
                                 M,
                                 pipe_name,
                                 idx_leak)

        case 'cha':
            rupture_ngs_pipe_cha(p,
                                 q,
                                 lam,
                                 va,
                                 D,
                                 S,
                                 dx,
                                 dt,
                                 M,
                                 pipe_name,
                                 idx_leak)
        case 'kt1':
            rupture_ngs_pipe_kt1(p,
                                 q,
                                 lam,
                                 va,
                                 D,
                                 S,
                                 dx,
                                 M,
                                 pipe_name,
                                 idx_leak)

        case 'kt2':
            rupture_ngs_pipe_kt2(p,
                                 q,
                                 lam,
                                 va,
                                 D,
                                 S,
                                 dx,
                                 M,
                                 pipe_name,
                                 idx_leak)
        case 'weno3':
            rupture_ngs_pipe_weno3(p,
                                   q,
                                   lam,
                                   va,
                                   D,
                                   S,
                                   dx,
                                   M,
                                   pipe_name,
                                   idx_leak)
        case _:
            raise NotImplementedError(f'No such method: {method}!')
    return artifact


def leakage_pipe():
    """
    Placeholder
    :return:
    """
    pass
