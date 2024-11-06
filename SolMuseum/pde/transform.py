from sympy import simplify

from Solverz.sym_algebra.symbols import Para, iAliasVar, IdxVar, iVar
from Solverz.sym_algebra.functions import switch
from Solverz.utilities.type_checker import is_number


class Pde(Eqn):
    """
    The class of partial differential equations
    """
    pass


class HyperbolicPde(Pde):
    r"""
    The class for hyperbolic PDE reading

    .. math::

         \frac{\partial{u}}{\partial{t}}+\frac{\partial{f(u)}}{\partial{x}}=S(u)

    where $u$ is the state vector, $f(u)$ is the flux function and $S(u)$ is the source term.

    Parameters
    ==========

    two_dim_var : iVar or list of Var

        Specify the two-dimensional variables in the PDE. Some of the variables, for example, the mass flow $\dot{m}$ in
        the heat transmission equation, are not two-dimensional variables.

    """

    def __init__(self, name: str,
                 diff_var: iVar | Var,
                 flux: Expr = 0,
                 source: Expr = 0,
                 two_dim_var: Union[iVar, Var, List[iVar | Var]] = None):
        if isinstance(source, (float, int)):
            source = sympify(source)
        super().__init__(name, source)
        diff_var = sVar2Var(diff_var)
        two_dim_var = sVar2Var(two_dim_var) if two_dim_var is not None else None
        self.diff_var = diff_var
        if isinstance(flux, (float, int)):
            flux = sympify(flux)
        if isinstance(source, (float, int)):
            flux = sympify(flux)
        self.flux = flux
        self.source = source
        self.two_dim_var = [two_dim_var] if isinstance(two_dim_var, iVar) else two_dim_var
        self.LHS = Derivative(diff_var, t) + Derivative(flux, x)

    def derive_derivative(self):
        pass

    def finite_difference(self, scheme='central diff', direction=None, M: int = 0, dx=None):
        r"""
        Discretize hyperbolic PDE as AEs.

        Parameters
        ==========

        scheme : str

            1 - Central difference

            .. math::

                \frac{\partial{u}}{\partial{t}}\approx\frac{u_{i+1}^{j+1}-u_{i+1}^{j}+u_{i}^{j+1}-u_{i}^{j}}{2\Delta t}

            .. math::

                \frac{\partial{f(u)}}{\partial{x}}\approx\frac{f(u_{i+1}^{j+1})-f(u_{i}^{j+1})+f(u_{i+1}^{j})-f(u_{i}^{j})}{2\Delta x}

            .. math::

                S(u)\approx S\left(\frac{u_{i+1}^{j+1}+u_{i}^{j+1}+u_{i+1}^{j}+u_{i}^{j}}{4}\right)

            2 - Backward Time Backward/Forward Space

            If direction equals 1, then do backward space difference, which derives

            .. math::

                \frac{\partial{u}}{\partial{t}}\approx\frac{u_{i+1}^{j+1}-u_{i+1}^{j}}{\Delta t}

            .. math::

                \frac{\partial{f(u)}}{\partial{x}}\approx\frac{f(u_{i+1}^{j+1})-f(u_{i}^{j+1})}{\Delta x}

            .. math::

                S(u)\approx S\left(u_{i+1}^{j+1}\right)

            If direction equals -1, then do forward space difference, which derives

            .. math::

                \frac{\partial{u}}{\partial{t}}\approx\frac{u_{i}^{j+1}-u_{i}^{j}}{\Delta t}

            .. math::

                \frac{\partial{f(u)}}{\partial{x}}\approx\frac{f(u_{i+1}^{j+1})-f(u_{i}^{j+1})}{\Delta x}

            .. math::

                S(u)\approx S\left(u_{i}^{j+1}\right)

        direction : int

            To tell which side of boundary conditions is given in scheme 2.

        M : int

        The total number of spatial sections.

        dx : Number

        Spatial difference step size

        Returns
        =======

        AE : Eqn

            Let's take central difference as an example, this function returns the algebraic equation

            .. math::

                \begin{aligned}
                    0=&\Delta x(\tilde{u}[1:M]-\tilde{u}^0[1:M]+\tilde{u}[0:M-1]-\tilde{u}^0[0:M-1])+\\
                      &\Delta t(f(\tilde{u}[1:M])-f(\tilde{u}[0:M-1])+f(\tilde{u}^0[1:M])-f(\tilde{u}^0[0:M-1]))+\\
                      &2\Delta x\Delta t\cdot S\left(\tilde{u}[1:M]-\tilde{u}^0[1:M]+\tilde{u}[0:M-1]-\tilde{u}^0[0:M-1]}{4}\right)
                \end{aligned}

            where we denote by vector $\tilde{u}$ the discrete spatial distribution of state $u$, by $\tilde{u}^0$ the
            initial value of $\tilde{u}$, and by $M$ the last index of $\tilde{u}$.

        """
        if isinstance(M, (int, np.integer)):
            if M < 0:
                raise ValueError(f'Total nunmber of PDE sections {M} < 0')
        else:
            raise TypeError(f'Do not support M of type {type(M)}')
        if M == 0:
            M = idx('M')

        if scheme == 'central diff':
            return Eqn('FDM of ' + self.name + 'w.r.t.' + self.diff_var.name + 'using central diff',
                       finite_difference(self.diff_var,
                                         self.flux,
                                         self.source,
                                         self.two_dim_var,
                                         M,
                                         'central diff',
                                         dx=dx))
        elif scheme == 'euler':
            return Eqn('FDM of ' + self.name + 'w.r.t.' + self.diff_var.name + 'using Euler',
                       finite_difference(self.diff_var,
                                         self.flux,
                                         self.source,
                                         self.two_dim_var,
                                         M,
                                         'euler',
                                         direction=direction,
                                         dx=dx))

    def semi_discretize(self,
                        a0=None,
                        a1=None,
                        scheme='TVD1',
                        M: int = 0,
                        output_boundary=True,
                        dx=None) -> List[Eqn]:
        r"""
        Semi-discretize the hyperbolic PDE of nonlinear conservation law as ODEs using the Kurganov-Tadmor scheme
        (see [Kurganov2000]_). The difference stencil is as follows, with $x_{j+1}-x_{j}=\Delta x$.

            .. image:: ../../pics/difference_stencil.png
               :height: 100

        Parameters
        ==========

        a0 : Expr

            Maximum local speed $a_{j+1/2}$, with formula

            .. math::

                a_{j+1/2}=\max\qty{\rho\qty(\pdv{f}{u}\qty(u^+_{j+1/2})),\rho\qty(\pdv{f}{u}\qty(u^-_{j+1/2}))},

            where

            .. math::

                \rho(A)=\max_i|\lambda_i(A)|.

            If $a_0$ or $a_1$ is None, then they will be set as ``Para`` ``ajp12`` and ``ajm12`` respectively.

        a1 : Expr

            Maximum local speed $a_{j-1/2}$, with formula

            .. math::

                a_{j-1/2}=\max\qty{\rho\qty(\pdv{f}{u}\qty(u^+_{j-1/2})),\rho\qty(\pdv{f}{u}\qty(u^-_{j-1/2}))}.

        scheme : str

            If scheme==1, 2nd scheme else, else, use 1st scheme.

        M : int

            The total number of spatial sections.

        output_boundary : bool

            If true, output equations about the boundary conditions. For example,

           >>> from Solverz import HyperbolicPde, iVar
           >>> T = iVar('T')
           >>> p = HyperbolicPde(name = 'heat transfer', diff_var=T, flux=T)
           >>> p.semi_discretize(a0=1,a2=1, scheme=2, M=2, output_boundary=True)
           1
           >>> p.semi_discretize(a0=1,a2=1, scheme=2, M=2, output_boundary=False)
           2

        dx : Number

            spatial difference step size

        Returns
        =======

        ODE : List[Union[Ode, Eqn]]

            This function returns the for $2\leq j\leq M-2$

            .. math::

                \dv{t}u_j=-\frac{H_{j+1/2}-H_{j-1/2}}{\Delta x}+S(u_j)

            and for $j=1,M-1$

            .. math::

                \dv{t}u_j=-\frac{f(u_{j+1})-f(u_{j-1})}{2\Delta x}+\frac{a_{j+1/2}(u_{j+1}-u_j)-a_{j-1/2}(u_j-u_{j-1})}{2\Delta x}+S(u_j),

            where

            .. math::

                H_{j+1/2}=\frac{f(u^+_{j+1/2})+f(u^-_{j+1/2})}{2}-\frac{a_{j+1/2}}{2}\qty[u^+_{j+1/2}-u^-_{j+1/2}],

            .. math::

                H_{j-1/2}=\frac{f(u^+_{j-1/2})+f(u^-_{j-1/2})}{2}-\frac{a_{j-1/2}}{2}\qty[u^+_{j-1/2}-u^-_{j-1/2}],

            .. math::

                u^+_{j+1/2}=u_{j+1}-\frac{\Delta x}{2}(u_x)_{j+1},\quad u^-_{j+1/2}=u_j+\frac{\Delta x}{2}(u_x)_j,

            .. math::

                u^+_{j-1/2}=u_{j}-\frac{\Delta x}{2}(u_x)_{j},\quad u^-_{j-1/2}=u_{j-1}+\frac{\Delta x}{2}(u_x)_{j-1},

            .. math::

                (u_x)_j=\operatorname{minmod}\qty(\theta\frac{u_j-u_{j-1}}{\Delta x},\frac{u_{j+1}-u_{j-1}}{2\Delta x},\theta\frac{u_{j+1}-u_{j}}{\Delta x}),\quad \theta\in[1,2],

            and by linear extrapolation

            .. math::

                u_0=2u_\text{L}-u_1,\quad u_M=2u_\text{R}-u_{M-1}.


        .. [Kurganov2000] Alexander Kurganov, Eitan Tadmor, New High-Resolution Central Schemes for Nonlinear Conservation Laws and Convection–Diffusion Equations, Journal of Computational Physics, Volume 160, Issue 1, 2000, Pages 241-282, `<https://doi.org/10.1006/jcph.2000.6459>`_

        """
        if isinstance(M, (int, np.integer)):
            if M < 0:
                raise ValueError(f'Total nunmber of PDE sections {M} < 0')
        else:
            raise TypeError(f'Do not support M of type {type(M)}')
        if M == 0:
            M = idx('M')
        u = self.diff_var
        dae_list = []
        if scheme == 'TVD2':
            eqn_dict = semi_descritize(self.diff_var,
                                       self.flux,
                                       self.source,
                                       self.two_dim_var,
                                       M,
                                       scheme='TVD2',
                                       a0=a0,
                                       a1=a1,
                                       dx=dx)
            dae_list.extend([Ode('SDM of ' + self.name + ' 1',
                                 eqn_dict['Ode'][0][0],
                                 eqn_dict['Ode'][0][1]),
                             Ode('SDM of ' + self.name + ' 2',
                                 eqn_dict['Ode'][1][0],
                                 eqn_dict['Ode'][1][1]),
                             Ode('SDM of ' + self.name + ' 3',
                                 eqn_dict['Ode'][2][0],
                                 eqn_dict['Ode'][2][1]),
                             Eqn('minmod limiter 1 of ' + u.name,
                                 eqn_dict['Eqn'][0]),
                             Eqn('minmod limiter 2 of ' + u.name,
                                 eqn_dict['Eqn'][1]),
                             Eqn('minmod limiter 3 of ' + u.name,
                                 eqn_dict['Eqn'][2])])
        elif scheme == 'TVD1':
            eqn_dict = semi_descritize(self.diff_var,
                                       self.flux,
                                       self.source,
                                       self.two_dim_var,
                                       M,
                                       scheme='TVD1',
                                       a0=a0,
                                       a1=a1,
                                       dx=dx)
            dae_list.extend([Ode('SDM of ' + self.name + 'using',
                                 eqn_dict['Ode'][0],
                                 eqn_dict['Ode'][1])])
        else:
            raise NotImplementedError(f'Scheme {scheme} not implemented')

        if output_boundary:
            dae_list.extend([Eqn(f'Equation of {u[M]}', u[M] - 2 * iVar(u.name + 'R') + u[M - 1]),
                             Eqn(f'Equation of {u[0]}', u[0] - 2 * iVar(u.name + 'L') + u[1])])

        return dae_list


def finite_difference(diff_var, flux, source, two_dim_var, M, scheme='central diff', direction=None, dx=None):
    M_ = M + 1  # for pretty printer of M in slice
    if dx is None:
        dx = Para('dx')
    else:
        if is_number(dx):
            dx = dx
        else:
            raise TypeError(f'Input dx is not number!')

    dt = Para('dt')
    u = diff_var
    u0 = iAliasVar(u.name + '_tag_0')

    if scheme == 'central diff':
        fui1j1 = flux.subs([(a, a[1:M_]) for a in two_dim_var])
        fuij1 = flux.subs([(a, a[0:M_ - 1]) for a in two_dim_var])
        fui1j = flux.subs([(a, iAliasVar(a.name + '_tag_0')[1:M_]) for a in two_dim_var])
        fuij = flux.subs([(a, iAliasVar(a.name + '_tag_0')[0:M_ - 1]) for a in two_dim_var])

        S = source.subs([(a, (a[1:M_] + a[0:M_ - 1] + iAliasVar(a.name + '_tag_0')[1:M_] + iAliasVar(a.name + '_tag_0')[
                                                                                           0:M_ - 1]) / 4) for a in
                         two_dim_var])

        fde = dx * (u[1:M_] - u0[1:M_] + u[0:M_ - 1] - u0[0:M_ - 1]) \
              + simplify(dt * (fui1j1 - fuij1 + fui1j - fuij)) \
              - simplify(2 * dx * dt * S)

    elif scheme == 'euler':
        if direction == 1:
            fui1j1 = flux.subs([(a, a[1:M_]) for a in two_dim_var])
            fuij1 = flux.subs([(a, a[0:M_ - 1]) for a in two_dim_var])
            S = source.subs([(a, a[1:M_]) for a in two_dim_var])
            fde = dx * (u[1:M_] - u0[1:M_]) + simplify(dt * (fui1j1 - fuij1)) - simplify(dx * dt * S)
        elif direction == -1:
            fui1j1 = flux.subs([(a, a[1:M_]) for a in two_dim_var])
            fuij1 = flux.subs([(a, a[0:M_ - 1]) for a in two_dim_var])
            S = source.subs([(a, a[0:M_ - 1]) for a in two_dim_var])
            fde = dx * (u[0:M_ - 1] - u0[0:M_ - 1]) + simplify(dt * (fui1j1 - fuij1)) - simplify(dx * dt * S)
        else:
            raise ValueError(f"Unimplemented direction {direction}!")
    return fde


def semi_descritize(diff_var,
                    flux,
                    source,
                    two_dim_var,
                    M,
                    scheme='TVD1',
                    a0=None,
                    a1=None,
                    dx=None):
    M_ = M + 1  # for pretty printer of M in slice

    if a0 is None:
        a0 = Para('ajp12')
    if a1 is None:
        a1 = Para('ajm12')

    if dx is None:
        dx = Para('dx')
    else:
        if is_number(dx):
            dx = dx
        else:
            raise TypeError(f'Input dx is not number!')

    u = diff_var
    if scheme == 'TVD2':
        # j=1
        # f(u[2])
        fu2 = flux.subs([(var, var[2]) for var in two_dim_var])
        # f(u[0])=f(2*uL-u[1])
        fu0 = flux.subs([(var, var[0]) for var in two_dim_var])
        # S(u[1])
        Su1 = source.subs([(var, var[1]) for var in two_dim_var])
        ode_rhs1 = -simplify((fu2 - fu0) / (2 * dx)) \
                   + simplify((a0[0] * (u[2] - u[1]) - a1[0] * (u[1] - u[0])) / (2 * dx)) \
                   + simplify(Su1)

        # j=M-1
        # f(u[M])=f(2*uR-u[M-1])
        fum = flux.subs([(var, var[M]) for var in two_dim_var])
        # f(u[M-2])
        fum2 = flux.subs([(var, var[M - 2]) for var in two_dim_var])
        # S(u[M-1])
        SuM1 = source.subs([(var, var[M - 1]) for var in two_dim_var])
        ode_rhs3 = -simplify((fum - fum2) / (2 * dx)) \
                   + simplify((a0[-1] * (u[M] - u[M - 1]) - a1[-1] * (u[M - 1] - u[M - 2])) / (2 * dx)) \
                   + simplify(SuM1)

        # 2<=j<=M-2
        def ujprime(U: IdxVar, v: int):
            # for given u_j,
            # returns
            # u^+_{j+1/2} case v==0,
            # u^-_{j+1/2} case 1,
            # u^+_{j-1/2} case 2,
            # u^-_{j-1/2} case 3
            if not isinstance(U.index, slice):
                raise TypeError("Index of IdxVar must be slice object")
            start = U.index.start
            stop = U.index.stop
            step = U.index.step
            U = U.symbol0
            Ux = iVar(U.name + 'x')

            # u_j
            Uj = U[start:stop:step]
            # (u_x)_j
            Uxj = Ux[start:stop:step]
            # u_{j+1}
            Ujp1 = U[start + 1:stop + 1:step]
            # (u_x)_{j+1}
            Uxjp1 = Ux[start + 1:stop + 1:step]
            # u_{j-1}
            Ujm1 = U[start - 1:stop - 1:step]
            # (u_x)_{j-1}
            Uxjm1 = Ux[start - 1:stop - 1:step]

            if v == 0:
                return Ujp1 - dx / 2 * Uxjp1
            elif v == 1:
                return Uj + dx / 2 * Uxj
            elif v == 2:
                return Uj - dx / 2 * Uxj
            elif v == 3:
                return Ujm1 + dx / 2 * Uxjm1
            else:
                raise ValueError("v=0 or 1 or 2 or 3!")

        # j\in [2:M_-2]
        Suj = source.subs([(var, var[2:M_ - 2]) for var in two_dim_var])
        Hp = (flux.subs([(var, ujprime(var[2:M_ - 2], 0)) for var in two_dim_var]) +
              flux.subs([(var, ujprime(var[2:M_ - 2], 1)) for var in two_dim_var])) / 2 \
             - a0[2:M_ - 2] / 2 * (ujprime(u[2:M_ - 2], 0) - ujprime(u[2:M_ - 2], 1))
        Hm = (flux.subs([(var, ujprime(var[2:M_ - 2], 2)) for var in two_dim_var]) +
              flux.subs([(var, ujprime(var[2:M_ - 2], 3)) for var in two_dim_var])) / 2 \
             - a1[2:M_ - 2] / 2 * (ujprime(u[2:M_ - 2], 2) - ujprime(u[2:M_ - 2], 3))
        ode_rhs2 = -simplify(Hp - Hm) / dx + Suj

        theta = Para('theta')
        ux = iVar(u.name + 'x')
        minmod_flag = Para('minmod_flag_of_' + ux.name)
        minmod_rhs = ux[1:M_ - 1] - switch(theta * (u[1:M_ - 1] - u[0:M_ - 2]) / dx,
                                           (u[2:M_] - u[0:M_ - 2]) / (2 * dx),
                                           theta * (u[2:M_] - u[1:M_ - 1]) / dx,
                                           0,
                                           minmod_flag)

        return {'Ode': [(ode_rhs1, u[1]), (ode_rhs2, u[2:M_ - 2]), (ode_rhs3, u[M - 1])],
                'Eqn': [minmod_rhs, ux[0], ux[M]]}
    elif scheme == 'TVD1':
        # 1<=j<=M-1
        # f(u[j+1])
        fu1 = flux.subs([(var, var[2:M_]) for var in two_dim_var])
        # f(u[j-1])
        fu2 = flux.subs([(var, var[0:M_ - 2]) for var in two_dim_var])
        # S(u[j])
        Su = source.subs([(var, var[1:M_ - 1]) for var in two_dim_var])
        ode_rhs = -simplify((fu1 - fu2) / (2 * dx)) \
                  + simplify((a0 * (u[2:M_] - u[1:M_ - 1]) - a1 * (u[1:M_ - 1] - u[0:M_ - 2])) / (2 * dx)) \
                  + simplify(Su)
        return {'Ode': (ode_rhs, u[1:M_ - 1])}
