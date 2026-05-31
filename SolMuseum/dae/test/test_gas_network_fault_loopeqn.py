"""Smoke + structural tests for the gas_network LoopEqn fault path.

The leakage-aware LoopEqn path packs each pipe's state into flat
``p_all`` / ``q_all`` Vars, with the leak cell promoted to an
algebraic tail entry and ``qleak1`` / ``qleak2`` algebraic variables
emitted per fault pipe. These tests confirm that:

* all four (loopeqn x with_fault) build paths compile through
  ``create_instance()``,
* the faulted ``loopeqn=True`` build exposes the bundled ``p_all`` /
  ``q_all`` Vars (not per-pipe ``p0..p9, q0..q9``) and emits the
  fault-pipe leakage boundary equations,
* the no-fault WENO3 LoopEqn path stays compact (8 grouped equation
  objects), and
* ``loopeqn=True`` raises a clear error for the not-yet-implemented
  rupture fault type instead of silently building a leakage model.

Long-horizon parity / step-count regression checks are the job of
the case-study benchmark scripts under
``case/case-small-sc/bench3way.py`` and ``case/case-min-ngs2/``.

The test loads the bundled IEGS case-small fixture
``fixtures/caseI.xlsx`` (vendored copy of the original
``caseI_2025_0423.xlsx``). The fixture lives in this directory so the
test suite is self-contained.
"""
import os

import pytest

from SolUtil import GasFlow
from SolMuseum.dae.gas_network import gas_network

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILE = os.path.join(HERE, 'fixtures', 'caseI.xlsx')


@pytest.fixture(scope='module')
def gf():
    if not os.path.exists(DATAFILE):
        pytest.skip(f'Test data not available at {DATAFILE}')
    flow = GasFlow(DATAFILE)
    flow.run()
    return flow


@pytest.fixture
def ng(gf):
    return gas_network(gf)


def _build(ng, loopeqn, with_fault, idx_leak=255, method='weno3', dt=None,
           fault_type='leakage'):
    if with_fault:
        return ng.mdl(dx=100, dt=dt, method=method,
                      fault_type=fault_type,
                      fault_pipe_index=[8],
                      fault_loc_index=[idx_leak],
                      leak_diameter=[0.5901 * 0.6],
                      loopeqn=loopeqn)
    return ng.mdl(dx=100, dt=dt, method=method, loopeqn=loopeqn)


def test_all_four_paths_build(ng):
    """no-fault and fault, x loopeqn and legacy — all must build."""
    for loopeqn in (True, False):
        for with_fault in (True, False):
            m = _build(ng, loopeqn, with_fault)
            eqs, y0 = m.create_instance()
            assert y0.array.size > 0


def test_loopeqn_fault_uses_bundled_layout(ng):
    """Faulted loopeqn=True exposes p_all / q_all + qleak1 / qleak2."""
    m = _build(ng, loopeqn=True, with_fault=True)
    assert 'p_all' in m.__dict__
    assert 'q_all' in m.__dict__
    assert 'p0' not in m.__dict__
    assert 'q0' not in m.__dict__
    assert 'p8' not in m.__dict__
    assert 'q8' not in m.__dict__


def test_legacy_fault_uses_per_pipe_layout(ng):
    """Faulted loopeqn=False keeps per-pipe Vars + per-pipe leak Eqns."""
    m = _build(ng, loopeqn=False, with_fault=True)
    assert 'p_all' not in m.__dict__
    assert 'q_all' not in m.__dict__
    for j in range(10):
        assert f'p{j}' in m.__dict__
        assert f'q{j}' in m.__dict__
    assert 'q_pipe8_leakage_leak1' in m.__dict__
    assert 'q_pipe8_leakage_leak2' in m.__dict__


def test_no_fault_loopeqn_path_unaffected(ng):
    """Regression guard: the no-fault LoopEqn build stays compact."""
    e, _ = _build(ng, loopeqn=True, with_fault=False).create_instance()
    # caseI: 11 nodes, 10 pipes, 1 slack ->
    # 8 LoopEqn groups (mass, p_inlet, p_outlet, bd_l, bd_r, gas_q,
    # gas_p, slack pressure as Eqn)
    assert len(e.EQNs) == 8


def test_loopeqn_rejects_unsupported_fault_type(ng):
    """rupture is not yet wired through the LoopEqn path."""
    with pytest.raises(NotImplementedError):
        _build(ng, loopeqn=True, with_fault=True, fault_type='rupture')


@pytest.mark.parametrize('method, dt', [
    ('kt1', None),
    ('cdm', 1.0),
    ('cha', 1.0),
    ('euler', 1.0),
])
def test_legacy_fault_supports_all_methods(ng, method, dt):
    """Per-pipe leakage discretizations all build via loopeqn=False."""
    m = _build(ng, loopeqn=False, with_fault=True, method=method, dt=dt)
    assert 'p8' in m.__dict__
    assert 'q8' in m.__dict__


@pytest.mark.parametrize('method', ['kt1', 'cdm', 'cha', 'euler', 'weno5'])
def test_loopeqn_rejects_non_weno3_method(ng, method):
    """loopeqn=True only honours weno3; any other scheme must raise rather
    than silently building a weno3 model (issue: method arg was a no-op)."""
    with pytest.raises(NotImplementedError):
        _build(ng, loopeqn=True, with_fault=False, method=method)
    with pytest.raises(NotImplementedError):
        _build(ng, loopeqn=True, with_fault=True, method=method)


def test_loopeqn_requires_M_at_least_2(ng):
    """floor(L/dx) < 2 collides bd_left / bd_right stencils against a
    neighbour's state; must raise (issue #7). caseI pipes are 51 km, so
    dx=30 km drives every pipe to M = floor(51000/30000) = 1."""
    with pytest.raises(ValueError, match=r'floor\(L/dx\) >= 2'):
        ng.mdl(dx=30000, loopeqn=True)


def test_loopeqn_rejects_compressor_type0(gf):
    """type-0 (compressor) pipes are silently dropped from mass continuity;
    the LoopEqn path must reject them up front (issue #6)."""
    ng2 = gas_network(gf)
    edge = next(iter(ng2.gf.gc['G'].edges(data=True)))
    saved = edge[2].get('type', 1)
    edge[2]['type'] = 0
    try:
        with pytest.raises(NotImplementedError, match='type-1'):
            ng2.mdl(dx=100, loopeqn=True)
    finally:
        edge[2]['type'] = saved


def test_loopeqn_rejects_nonzero_delta(gf):
    """A non-zero compressor delta has no boost term in the dynamic model;
    reject it rather than ignore it (issue #6)."""
    import numpy as np
    ng2 = gas_network(gf)
    saved = np.asarray(ng2.gf.delta).copy()
    ng2.gf.delta = saved.copy()
    ng2.gf.delta[0] = 1.0
    try:
        with pytest.raises(NotImplementedError, match='delta'):
            ng2.mdl(dx=100, loopeqn=True)
    finally:
        ng2.gf.delta = saved
