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
    The rupture fault model of ngs pipes, with the fault boundary condition

        .. math::
            \begin{aligned}
            &p_i = p_a\\
            &q_\text{i} = q_\text{upstream} - q_\text{downstream}
            \end{aligned}

    where $p_i$ is the pressure of the rupture position, $p_a$ is the atmospheric pressure, $q_i$ is the mass flow of
    the rupture position, $q_\text{upstream}$ is the upstream leakage mass flow and $q_\text{downstream}$ is the downstream
    leakage mass flow.
    The fault model is based on the normal ngs pipe defined above. The details can be found in [1]_.

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

    References
    ==========

    .. [1] https://arxiv.org/abs/2410.09464

    """

    if not is_integer(M):
        raise TypeError(f'M is {type(M)} instead of integer')

    if not is_number(dx):
        raise TypeError(f'dx is {type(dx)} instead of number')

    if not is_number(idx_leak):
        raise TypeError(f'idx_leak is {type(idx_leak)} instead of number')

    match method:
        case 'euler':
            return rupture_ngs_pipe_Euler(p,
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
            return rupture_ngs_pipe_cdm(p,
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
            return rupture_ngs_pipe_cha(p,
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
            return rupture_ngs_pipe_kt1(p,
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
            return rupture_ngs_pipe_kt2(p,
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
            return rupture_ngs_pipe_weno3(p,
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


def leakage_pipe(p: Var,
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
                 d,
                 method='weno3'):
    r"""
    The leakage fault model of ngs pipes, with the fault boundary condition

        .. math::
            q_\text{leak}=
            \begin{cases}
                0.61S_\text{h}p_j\sqrt{\frac{2M}{ZRT}\frac{k}{k-1}\left(\left(\frac{p_\text{a}}{p_j}\right)^{\frac{2}{k}}-\left(\frac{p_\text{a}}{p_j}\right)^{\frac{k+1}{k}}\right)}&p_j\leq {p_\text{sw}}\\
                S_\text{h}p_j\sqrt{\frac{M}{ZRT}k\left(\frac{2}{k+1}\right)^{\frac{k+1}{k-1}}} & p_j > {p_\text{sw}}\\
            \end{cases}

    $k$ is the adiabatic coefficient, $M$ is the molar mass of gas, $T$ is 273.15K, $Z$ is the gas
    compressibility factor, $R$ is the ideal gas constant, $S_\text{h}$ is the area of the leakage hole;

        .. math::
            S_\text{h} = \pi\qty(\frac{d}{2})^2,

    $d$ is the diameter of the leakage hole; the switching pressure

        .. math::
            p_\text{sw}=\left(\frac{2}{k+1}\right)^{-k/(k-1)}\cdot p_{a}.

    The fault model is based on the normal ngs pipe defined above.

    Parameters
    ==========

    idx_leak: Number

        The index of rupture location.

    d: Number or Param

        The diameter of the leakage hole.

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

    match method:
        case 'euler':
            return leakage_ngs_pipe_Euler(p,
                                          q,
                                          lam,
                                          va,
                                          D,
                                          S,
                                          dx,
                                          dt,
                                          M,
                                          pipe_name,
                                          idx_leak,
                                          d)

        case 'cdm':
            return leakage_ngs_pipe_cdm(p,
                                        q,
                                        lam,
                                        va,
                                        D,
                                        S,
                                        dx,
                                        dt,
                                        M,
                                        pipe_name,
                                        idx_leak,
                                        d)

        case 'cha':
            return leakage_ngs_pipe_cha(p,
                                        q,
                                        lam,
                                        va,
                                        D,
                                        S,
                                        dx,
                                        dt,
                                        M,
                                        pipe_name,
                                        idx_leak,
                                        d)
        case 'kt1':
            return leakage_ngs_pipe_kt1(p,
                                        q,
                                        lam,
                                        va,
                                        D,
                                        S,
                                        dx,
                                        M,
                                        pipe_name,
                                        idx_leak,
                                        d)

        case 'kt2':
            return leakage_ngs_pipe_kt2(p,
                                        q,
                                        lam,
                                        va,
                                        D,
                                        S,
                                        dx,
                                        M,
                                        pipe_name,
                                        idx_leak,
                                        d)
        case 'weno3':
            return leakage_ngs_pipe_weno3(p,
                                          q,
                                          lam,
                                          va,
                                          D,
                                          S,
                                          dx,
                                          M,
                                          pipe_name,
                                          idx_leak,
                                          d)
        case _:
            raise NotImplementedError(f'No such method: {method}!')
