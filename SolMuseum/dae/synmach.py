from Solverz import Eqn, Ode, Param, Model, cos, sin
from Solverz import Var
from Solverz.utilities.type_checker import is_number
import numpy as np
from SolMuseum.util import rename_mdl


class synmach:

    def __init__(self,
                 name='syn',
                 use_coi=True,
                 ws=2 * np.pi * 50,
                 **kwargs):
        self.name = name
        self.use_coi = use_coi

        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = np.array(value)
            elif is_number(value):
                kwargs[key] = np.array(value).reshape((-1,))

        self.ux = kwargs.get('ux')
        self.uy = kwargs.get('uy')
        self.ix = kwargs.get('ix')
        self.iy = kwargs.get('iy')

        self.ra = kwargs.get('ra')
        self.xdp = kwargs.get('xdp')
        self.xqp = kwargs.get('xqp')
        self.xq = kwargs.get('xq')
        self.Damping = kwargs.get('Damping')
        self.Tj = kwargs.get('Tj')
        self.ws = ws

        self.n_mac = len(self.Tj)

        self.omega = None
        self.delta = None
        self.Edp = None
        self.Eqp = None
        self.Pe = None
        self.Pm = None

    def synmach_init(self):
        u = self.ux + 1j * self.uy
        i = self.ix + 1j * self.iy
        self.delta = np.angle(u + (self.ra + 1j * self.xq) * i)
        udq = u * np.exp(-1j * (self.delta - np.pi / 2))
        idq = i * np.exp(-1j * (self.delta - np.pi / 2))
        ud = udq.real
        uq = udq.imag
        id_syn = idq.real
        iq_syn = idq.imag
        self.Edp = ud + self.ra * id_syn - self.xqp * iq_syn
        self.Eqp = uq + self.ra * iq_syn + self.xdp * id_syn
        self.Pe = (ud + self.ra * id_syn) * id_syn + (uq + self.ra * iq_syn) * iq_syn
        self.Pm = self.Pe
        self.omega = np.ones_like(self.delta)

    def mdl(self,
            rename=True):
        self.synmach_init()

        m = Model()

        name = self.name

        m.ux = Var('ux_' + name, self.ux)
        m.uy = Var('uy_' + name, self.uy)
        m.ix = Var('ix_' + name, self.ix)
        m.iy = Var('iy_' + name, self.iy)
        m.delta = Var('delta_' + name, self.delta)
        m.omega = Var('omega_' + name, self.omega)

        # if not self.pm_is_var:
        #     m.Pm = TimeSeriesParam('Pm_' + name,
        #                            [self.Pe[0], self.Pe[0]],
        #                            [0, 1000])
        # else:
        #     m.Pm = Var('Pm_' + name, self.Pe)
        m.Pm = Var('Pm_' + name, self.Pm)

        m.Damping = Param('Damping_' + name, self.Damping)
        m.Tj = Param('Tj_' + name, self.Tj)
        m.Edp = Param('Edp_' + name, self.Edp)
        m.Eqp = Param('Eqp_' + name, self.Eqp)
        m.ra = Param('ra_' + name, self.ra)
        m.xdp = Param('xdp_' + name, self.xdp)
        m.xqp = Param('xqp_' + name, self.xqp)

        Pe = m.ux * m.ix + m.uy * m.iy + (m.ix ** 2 + m.iy ** 2) * m.ra
        m.rotator_eqn = Ode(name='rotator_speed' + name,
                            f=(m.Pm - Pe - m.Damping * (m.omega - 1)) / m.Tj,
                            diff_var=m.omega)
        if self.use_coi:
            omega_coi = Var('omega_coi', 1)
        else:
            omega_coi = 1

        m.delta_eq = Ode(f'Delta_equation' + name,
                         self.ws * (m.omega - omega_coi),
                         diff_var=m.delta)
        m.Ed_prime = Eqn(name='Ed_prime' + name,
                         eqn=(m.Edp - sin(m.delta) * (m.ux + m.ra * m.ix - m.xqp * m.iy)
                              + cos(m.delta) * (m.uy + m.ra * m.iy + m.xqp * m.ix)))
        m.Eq_prime = Eqn(name='Eq_prime' + name,
                         eqn=(m.Eqp - cos(m.delta) * (m.ux + m.ra * m.ix - m.xdp * m.iy)
                              - sin(m.delta) * (m.uy + m.ra * m.iy + m.xdp * m.ix)))

        if rename:
            m = rename_mdl(m, name)

        return m
