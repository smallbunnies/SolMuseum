import os
import tempfile
import uuid

from Solverz import Eqn, module_printer
from SolMuseum._version import __version__ as sm_ver
from SolMuseum.dae.synmach import synmach


def _render_init(model):
    spf, y0 = model.create_instance()
    d = tempfile.mkdtemp(prefix='sm_prov_')
    name = f'prov_{uuid.uuid4().hex[:8]}'
    module_printer(spf, y0, name, directory=d, jit=False).render()
    with open(os.path.join(d, name, '__init__.py')) as f:
        return f.read()


def _build_synmach():
    """Smallest standalone SolMuseum component: a synchronous machine
    built from plain scalars (mirrors test_syn.py's fixture)."""
    return synmach(ux=1.0607541902861368,
                   uy=0.175893046755461,
                   ix=0.503496235830359,
                   iy=0.2988361191522479,
                   ra=0,
                   xdp=0.0608,
                   xqp=0.0969,
                   xq=0.0969,
                   Damping=100,
                   Tj=47.28,
                   use_coi=False)


def test_synmach_stamps_solmuseum_version_and_component():
    mi = _build_synmach()
    m = mi.mdl()
    # Close the network-facing terminal: pin bus voltage and Pm so the
    # standalone machine model is square and renderable (mirrors
    # test_syn.py). These three closing Eqns are user-defined and stay
    # unsourced; the synmach-built equations must carry the SolMuseum stamp.
    m.eqn_ux_syn = Eqn('ux_syn', m.ux_syn - 1.0607541902861368)
    m.eqn_uy_syn = Eqn('uy_syn', m.uy_syn - 0.175893046755461)
    m.eqn_pm_syn = Eqn('pm_syn', m.Pm_syn - mi.Pm[0])

    init = _render_init(m)
    assert f'SolMuseum {sm_ver}' in init
    assert 'synmach' in init
