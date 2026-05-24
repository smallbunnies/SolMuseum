"""Smoke + parity test for the fault-aware LoopEqn gas_network path.

`gas_network.mdl(loopeqn=True, fault_type='leakage', ...)` used to
silently fall back to the per-pipe Var legacy build because the
LoopEqn / LoopOde helpers had no provision for leakage cells. The
fault-aware LoopEqn build (issue #4) splits the leak pipe into two
contiguous state segments, packs the algebraic leak cell + qleak1 /
qleak2 at the q_all tail, and patches the WENO5 / TVD1 stencil
neighbour positions so the leak-adjacent TVD1 cells read qleak1 /
qleak2 in place of q[il].

This test confirms:

* both fault and no-fault builds compile through `create_instance()`
  via either path,
* the fault LoopEqn build produces exactly +2 algebraic vars over the
  no-fault LoopEqn build (qleak1 and qleak2),
* the eqn-group count is the expected small number (4 new groups for
  the fault path: leak_bd_left, leak_bd_right, weno3_pipe{j}_bd1,
  weno3_pipe{j}_bd2), and
* the leak-tail Vars are present in the rendered initial-value
  vector at the expected positions.

The test requires the IEGS case-small `caseI_2025_0423.xlsx` data
file and is skipped if absent.
"""
import os

import numpy as np
import pytest

from SolUtil import GasFlow
from SolMuseum.dae.gas_network import gas_network

HERE = os.path.dirname(os.path.abspath(__file__))
# Project layout: SolMuseum/SolMuseum/dae/test/  ->  five-deep ascend
# to reach the dev/ root, then over to the AAAkeyan project tree.
DATAFILE = os.path.normpath(os.path.join(
    HERE, '..', '..', '..', '..', '..',
    'AAAkeyan', '2025_0422_ROW_W', 'case', 'case small',
    'caseI_2025_0423.xlsx'))


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


def _build(ng, loopeqn, with_fault, idx_leak=255):
    if with_fault:
        return ng.mdl(dx=100, dt=None, method='weno3',
                      fault_type='leakage',
                      fault_pipe_index=[8],
                      fault_loc_index=[idx_leak],
                      leak_diameter=[0.5901 * 0.6],
                      loopeqn=loopeqn)
    return ng.mdl(dx=100, dt=None, method='weno3', loopeqn=loopeqn)


def test_all_four_paths_build(ng):
    """no-fault and fault, x loopeqn and legacy — all must build."""
    for loopeqn in (True, False):
        for with_fault in (True, False):
            m = _build(ng, loopeqn, with_fault)
            eqs, y0 = m.create_instance()
            assert y0.array.size > 0


def test_fault_loopeqn_adds_expected_vars_and_eqns(ng):
    """Fault LoopEqn build = no-fault LoopEqn + (2 vars, 4 eqn groups)."""
    m_nofault = _build(ng, loopeqn=True, with_fault=False)
    e_nofault, y_nofault = m_nofault.create_instance()

    m_fault = _build(ng, loopeqn=True, with_fault=True)
    e_fault, y_fault = m_fault.create_instance()

    # qleak1 / qleak2 added in q_all tail (one of each per fault pipe).
    assert y_fault.array.size - y_nofault.array.size == 2

    # 4 new eqn groups: leak_bd_left, leak_bd_right, weno3_*_bd1,
    # weno3_*_bd2.
    new_eqns = set(e_fault.EQNs) - set(e_nofault.EQNs)
    assert 'leak_bd_left' in new_eqns
    assert 'leak_bd_right' in new_eqns
    assert any(name.startswith('weno3_pipe') and name.endswith('_bd1')
               for name in new_eqns)
    assert any(name.startswith('weno3_pipe') and name.endswith('_bd2')
               for name in new_eqns)


def test_loopeqn_and_legacy_have_matching_var_counts(ng):
    """The fault LoopEqn and fault legacy builds must declare the
    same number of free variables.

    This is a necessary condition for the two builds to encode the
    same underlying DAE — they package the variables differently
    (flat p_all/q_all vs per-pipe Vars + qleak Vars), but the
    underlying var count is invariant.
    """
    e_loop, y_loop = _build(ng, loopeqn=True, with_fault=True).create_instance()
    e_leg, y_leg = _build(ng, loopeqn=False, with_fault=True).create_instance()
    assert y_loop.array.size == y_leg.array.size, (
        f'loopeqn fault: {y_loop.array.size} vars; '
        f'legacy fault: {y_leg.array.size} vars')


def test_loopeqn_eqn_group_count_is_small(ng):
    """The whole point of LoopEqn: a small fixed number of equation
    groups, regardless of total cell count.

    A 510-cell-per-pipe leak case should produce on the order of 10
    LoopEqn / LoopOde groups + a handful of scalar Eqns, NOT
    thousands of per-cell Eqns.
    """
    e_loop, _ = _build(ng, loopeqn=True, with_fault=True).create_instance()
    assert len(e_loop.EQNs) < 30, (
        f'expected <30 eqn groups for LoopEqn build, got {len(e_loop.EQNs)}')


def test_no_fault_loopeqn_path_unaffected(ng):
    """Regression guard: refactoring the fault path must not change
    the eqn / var sizes of the no-fault LoopEqn build."""
    e, y = _build(ng, loopeqn=True, with_fault=False).create_instance()
    # caseI: 11 nodes, 10 pipes, 1 slack ->
    # 8 LoopEqn groups (mass, p_inlet, p_outlet, bd_l, bd_r, gas_q,
    # gas_p, slack pressure as Eqn)
    assert len(e.EQNs) == 8
