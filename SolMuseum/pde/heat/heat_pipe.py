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
              method='kt2'):
    r"""
    This function model and discretize natural gas equations

        .. math::
            \pdv{p}{t}+\frac{c^2}{S}\pdv{q}{x}=&0,\\
            \pdv{q}{t}+S\pdv{p}{x}+\frac{\lambda c^2q|q|}{2DSp}=&0

    Parameters
    ==========

    p : Var
        The pressure w.r.t. x and t

    q : Var
        The mass flow w.r.t. x and t

    lam : Param or Number
        The friction $\lambda$

    va : Param or Number
        The speed of sound

    D : Param or Number
        The pipe diameter

    S : Param or Number
        The cross-section area

    dx : Param or Number
        The spatial difference step size

    M : Param or Number
        The friction $\lambda$

    pipe_name : str
        The name of the pipe

    method : str

        'cha' - The method of characteristics

            .. math ::
                p_i^{j+1}-p_{i-1}^j+\frac{c}{S}\left(q_i^{j+1}-q_{i-1}^j\right)+\frac{\lambda c^2 \Delta x}{4 D S^2} \frac{\left(q_i^{j+1}+q_{i-1}^j\right)\left|q_i^{j+1}+q_{i-1}^j\right|}{p_i^{j+1}+p_{i-1}^j}&=0,\quad 1\leq i\leq M,\\
                p_{i+1}^j-p_i^{j+1}+\frac{c}{S}\left(q_i^{j+1}-q_{i+1}^j\right)+\frac{\lambda c^2 \Delta x}{4 D S^2} \frac{\left(q_i^{j+1}+q_{i+1}^j\right)\left|q_i^{j+1}+q_{i+1}^j\right|}{p_i^{j+1}+p_{i+1}^j}&=0,\quad 0\leq i\leq M-1.

        'weno' - The WENO semi-discretization

            .. math ::
                \pdv{u_j}{t}=-\frac{1}{\Delta x}\qty(\hat{f}_{j+1/2}-\hat{f}_{j-1/2})+S(u_j)

            where $\hat{f}_{j+1/2}$ and $\hat{f}_{j-1/2}$ are reconstructed by the weighted essentially non-oscillatory
            scheme [1]_.

    Returns
    =======

    artifact : dict()
        The dict of derived equations and variables.

    References
    ==========

    .. [1] C.-W. Shu, “Essentially non-oscillatory and weighted essentially non- oscillatory schemes for hyperbolic conservation laws,” in Advanced Numerical Approximation of Nonlinear Hyperbolic Equations: Lectures given at the 2nd Session of the Centro Internazionale Matematico Estivo (C.I.M.E.) held in Cetraro, Italy, June 23–28, 1997, A. Quarteroni, Ed. Berlin, Heidelberg: Springer Berlin Heidelberg, 1998, pp. 325–432.

    """

    if not is_integer(M):
        raise TypeError(f'M is {type(M)} instead of integer')

    if not is_number(dx):
        raise TypeError(f'dx is {type(dx)} instead of number')

    artifact = dict()

    match method:
        case 'kt2':
            rhs = mol_kt2_rhs(T,
                              m,
                              lam,
                              rho,
                              Cp,
                              S,
                              Tamb,
                              dx,
                              1,
                              2,
                              order=1)
            artifact['T' + pipe_name + '_eqn1'] = Ode(f'heat_pipe_kt2_T{pipe_name}_1',
                                                      rhs,
                                                      T[1])
            rhs = mol_kt2_rhs(T,
                              m,
                              lam,
                              rho,
                              Cp,
                              S,
                              Tamb,
                              dx,
                              2,
                              M,
                              order=2)
            artifact['T' + pipe_name + '_eqn2'] = Ode(f'heat_pipe_kt2_T{pipe_name}_2',
                                                      rhs,
                                                      T[2:M])
            rhs = mol_kt2_rhs(T,
                              m,
                              lam,
                              rho,
                              Cp,
                              S,
                              Tamb,
                              dx,
                              M,
                              M+1,
                              order=1)
            artifact['T' + pipe_name + '_eqn3'] = Ode(f'heat_pipe_kt2_T{pipe_name}_3',
                                                      rhs,
                                                      T[M])
            artifact['theta'] = Param('theta', 1)
        case _:
            raise NotImplementedError(f'No such method: {method}!')
    return artifact
