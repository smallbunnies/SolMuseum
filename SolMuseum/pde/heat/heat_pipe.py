from .util import *


def heat_pipe(T: Var,
              m,
              lam,
              rho,
              Cp,
              S,
              Tamb,
              dx,
              dt,
              M,
              pipe_name: str,
              method='kt2',
              T_offset: int = 0):
    r"""
    This function model and discretize heat pipe equations

        .. math::
            \frac{\partial\tau}{\partial t}+\frac{\dot{m}}{\gamma \rho}\frac{\partial\tau}{\partial x}+
            \frac{\lambda }{\gamma\rho C_\mathrm{p}}(\tau-\tau^\mathrm{amb})=0

    where $\tau$ denotes the two-dimensional temperature distribution, $\gamma$ is the cross sectional ares, $\rho$ is
    the water density, $\dot{m}$ is the water mass flow, $\lambda$ is the friction coefficient, $C_p$ is the thermal
    capacity, and $\tau^\text{amb}$ is the ambient temperature.

    Parameters
    ==========

    T : Var
        The temperature distribution w.r.t. x and t

    m: Param or Var
        The mass flow rate

    lam : Param or Number
        The friction $\lambda$

    rho: Param or Number
        The water density

    Cp: Param or Number
        Thermal capacity of mass flow

    S : Param or Number
        The cross-section area

    Tamb: Param or Number
        The ambient temperature

    dx : Param or Number
        The spatial difference step size

    dt : Param or Number
        The temporal step size. `dt` can be set to 0 if one uses the kt2 scheme.

    M : Param or Number
        The friction $\lambda$

    pipe_name : str
        The name of the pipe

    method : str

        'iu' - The method of implicit upwind [1]_

            .. math ::
                \left\{
                \begin{aligned}
                    &\frac{\partial \tau}{\partial t}=\frac{\tau_{k+1}^{n+1}-\tau_{k+1}^{n}}{\Delta t}\\
                    &\frac{\partial \tau}{\partial x}=\frac{\tau_{k+1}^{n+1}-\tau_k^{n+1}}{\Delta x}\\
                    &\tau=\tau_{k+1}^{n+1}
                \end{aligned}
                \right.

        'yao' - The Yao's scheme, i.e. the second order explicit scheme [2]_

            .. math ::
                \left\{
                \begin{aligned}
                    &\frac{\partial \tau}{\partial t}=\frac{\tau_k^{n+1}-\tau_k^{n}+\tau_{k+1}^{n+1}-\tau_{k+1}^{n}}{2\Delta t}\\
                    &\frac{\partial \tau}{\partial x}=\frac{\tau_{k+1}^{n+1}-\tau_k^{n+1}+\tau_{k+1}^{n}-\tau_{k}^{n}}{2\Delta x}\\
                     &\tau=\frac{\tau_{k+1}^{n+1}+\tau_k^{n+1}+\tau_{k+1}^{n}+\tau_{k}^{n}}{4}
                \end{aligned}
                \right.

        'kt2' - Default, the second order kurganov-tadmor scheme [3]_

            .. math ::
                \pdv{u_j}{t}=-\frac{1}{\Delta x}\qty(\hat{f}_{j+1/2}-\hat{f}_{j-1/2})+S(u_j)

            where $\hat{f}_{j+1/2}$ and $\hat{f}_{j-1/2}$ are reconstructed by the second order kurganov-tadmor scheme.

    Returns
    =======

    artifact : dict()
        The dict of derived equations and variables.

    References
    ==========

    .. [1] https://doi.org/10.1016/j.apenergy.2017.08.061

    .. [2] https://doi.org/10.1109/TSTE.2020.2988682

    .. [3] https://doi.org/10.1006/jcph.2000.6459

    """

    if not is_integer(M):
        raise TypeError(f'M is {type(M)} instead of integer')

    if not is_number(dx):
        raise TypeError(f'dx is {type(dx)} instead of number')

    if not is_integer(T_offset):
        raise TypeError(f'T_offset is {type(T_offset)} instead of integer')
    off = int(T_offset)

    artifact = dict()

    if method not in ['kt2'] and dt == 0:
        raise ValueError('dt must be greater than 0!')

    match method:
        case 'kt2':
            if M < 3:
                raise ValueError(
                    f"M must be at least 3 for kt2 discretization scheme, but got M={M} "
                    f"for pipe '{pipe_name}'. Consider using a smaller dx value"
                )

            theta = Param('theta', 1)
            rhs = kt1_ode(T[off+0:off+1], T[off+1:off+2], m, lam, rho, Cp, S, Tamb, dx)
            artifact['T' + pipe_name + '_eqn1'] = Ode(f'heat_pipe_kt2_T{pipe_name}_1',
                                                      rhs,
                                                      T[off+1:off+2])

            rhs = kt2_ode(T[off+0:off+M-2], T[off+1:off+M-1], T[off+2:off+M], T[off+3:off+M+1],
                          m, lam, rho, Cp, S, Tamb, theta, dx)
            artifact['T' + pipe_name + '_eqn2'] = Ode(f'heat_pipe_kt2_T{pipe_name}_2',
                                                      rhs,
                                                      T[off+2:off+M])
            rhs = kt1_ode(T[off+M-1:off+M], T[off+M:off+M+1], m, lam, rho, Cp, S, Tamb, dx)
            artifact['T' + pipe_name + '_eqn3'] = Ode(f'heat_pipe_kt2_T{pipe_name}_3',
                                                      rhs,
                                                      T[off+M:off+M+1])
            artifact['theta'] = Param('theta', 1)
        case 'iu':
            if is_number(dt):
                if dt <= 0:
                    raise ValueError(f'dt must be positive, got {dt}')
            T0 = AliasVar(T.name, init=T)
            artifact[T0.name] = T0
            dTdt = (T[off+1:off+M+1]-T0[off+1:off+M+1])/dt
            dTdx = (T[off+1:off+M+1]-T[off+0:off+M])/dx
            rhs = dTdt + m/S/rho * dTdx + lam/S/rho/Cp*(T[off+1:off+M+1]-Tamb)
            artifact[f'iu_pipe_{pipe_name}'] = Eqn(f'iu_pipe_{pipe_name}', rhs)
        case 'yao':
            if is_number(dt):
                if dt <= 0:
                    raise ValueError(f'dt must be positive, got {dt}')
            T0 = AliasVar(T.name, init=T)
            artifact[T0.name] = T0
            dTdt = (T[off+1:off+M+1]-T0[off+1:off+M+1]+T[off+0:off+M]-T0[off+0:off+M])/(2*dt)
            dTdx = (T[off+1:off+M+1]+T0[off+1:off+M+1]-T[off+0:off+M]-T0[off+0:off+M])/(2*dx)
            Tavg = (T[off+1:off+M+1]+T0[off+1:off+M+1]+T[off+0:off+M]+T0[off+0:off+M])/4
            rhs = dTdt + m/S/rho*dTdx + lam/S/rho/Cp*(Tavg-Tamb)
            artifact[f'yao_pipe_{pipe_name}'] = Eqn(f'yao_pipe_{pipe_name}', rhs)
        case _:
            raise NotImplementedError(f'No such method: {method}!')
    return artifact
