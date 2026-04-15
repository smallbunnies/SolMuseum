"""LoopOde-based wrapper over the classic ``heat_pipe`` PDE semantics.

``heat_pipe`` from ``heat_pipe.py`` is the authoritative per-pipe
kt2 / iu / yao discretizer. Its public contract is stable and
external callers depend on its exact signature and behaviour —
see e.g. ``SolMuseum.dae.fault_heat_network``. This module adds a
parallel ``heat_pipe_loop`` function that is **structurally** the
same as ``heat_pipe(method='kt2')`` but, instead of emitting 3
``Ode``s per pipe (head cell + interior cells + tail cell), it
collapses the stencil into a single :class:`Solverz.LoopOde`.

The generated module gets ONE ``inner_F<N>`` per call with a
Python ``for`` loop over the pipe's M state cells, replacing 3
straight-line ``inner_F<N>``s per pipe (6 per pipe if you count
both supply and return). For an IES benchmark with 11 pipes × 2
sides × 3 Odes = 66 per-cell ``inner_F<N>``s, the LoopOde variant
cuts this to 22 (11 × 2 sides × 1 LoopOde) — one for-loop kernel
per pipe-side.

``heat_pipe`` itself is imported below solely so consumers of this
module can fall back to it under non-kt2 methods without re-
importing. ``heat_pipe_loop`` does NOT modify, monkey-patch, or
extend ``heat_pipe`` in any way.
"""
import sympy as sp
from sympy.functions.special.tensor_functions import KroneckerDelta

from Solverz import Idx, LoopOde, Param, Var
from Solverz.utilities.type_checker import is_integer, is_number

from .kt1.kt1_pipe import kt1_ode
from .kt2.kt2_pipe import kt2_ode


def heat_pipe_loop(T: Var,
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
                    model,
                    T_offset: int = 0):
    r"""Build a single-``LoopOde`` kt2 discretization of one pipe.

    Same semantics as ``heat_pipe(method='kt2')`` — the resulting
    DAE is algebraically equivalent — but the generated module emits
    ONE ``inner_F<N>`` with a for loop instead of 3 per-cell
    ``inner_F<N>``s.

    Parameters
    ----------
    T : Solverz Var
        The (flat) temperature Var that this pipe's cells live in.
        Usually ``m.Tsp_all`` or ``m.Trp_all`` from
        ``heat_network._mdl_loopeqn``.
    m, lam, rho, Cp, S, Tamb : Solverz Var / Param / IdxVar / IdxPara
        Exactly the same as ``heat_pipe``'s arguments. ``m``,
        ``lam``, ``S`` are typically per-pipe scalar ``IdxVar`` /
        ``IdxPara`` (e.g. ``m.m[j]``); ``rho``, ``Cp``, ``Tamb`` are
        scalar Params.
    dx : float
        Spatial step.
    dt : float
        Temporal step (unused for kt2 but present for signature
        parity with ``heat_pipe``).
    M : int
        Number of spatial cells for this pipe (state cells =
        cells 1..M; cell 0 is BC-pinned).
    pipe_name : str
        Unique suffix for generated Eqn / Ode names.
    model : Solverz Model
        The containing model. Required so :class:`LoopOde` can
        auto-resolve Solverz ``IdxVar`` / ``IdxPara`` references in
        the body. ``heat_pipe`` doesn't need this because its body
        goes through plain ``Ode`` lambdify which handles
        ``IdxVar`` via Solverz's standard rewriter.
    T_offset : int, default 0
        Offset into ``T`` where this pipe's segment starts. Matches
        the ``T_offset`` convention ``heat_pipe`` itself uses.

    Returns
    -------
    artifact : dict
        ``{'T{pipe_name}_loopode': LoopOde(...), 'theta': Param}``.
        The caller ``m.add(...)``s it the same way as
        ``heat_pipe(...)``'s output.
    """
    if not is_integer(M):
        raise TypeError(f'M is {type(M)} instead of integer')
    if not is_number(dx):
        raise TypeError(f'dx is {type(dx)} instead of number')
    if not is_integer(T_offset):
        raise TypeError(f'T_offset is {type(T_offset)} instead of integer')
    if M < 3:
        raise ValueError(
            f"M must be at least 3 for kt2 loop discretization, "
            f"but got M={M} for pipe '{pipe_name}'. Consider using "
            f"a smaller dx value"
        )

    off = int(T_offset)
    artifact = dict()

    # ``theta`` is a per-model-level singleton the kt2 stencil needs.
    # Register it on the model lazily so the LoopOde body walker can
    # resolve it from ``model.theta`` at ``_rewrite_solverz_body``
    # time. If the user (or a previous heat_pipe_loop call) already
    # attached it, leave it alone.
    if not hasattr(model, 'theta'):
        model.theta = Param('theta', 1)
    theta_sym = model.theta
    artifact['theta'] = Param('theta', 1)  # mirrors heat_pipe behaviour

    # Sympy IndexedBase over the flat T Var name. LoopOde's body
    # walker calls ``_rewrite_solverz_body`` (via LoopEqn.__init__
    # with ``model=...``) which recognizes ``sp.IndexedBase``
    # references by looking up ``model.<base_name>`` automatically.
    # Using IndexedBase instead of ``T[off + k]`` (Solverz
    # ``Var.__getitem__``) sidesteps the fact that Solverz Vars don't
    # currently accept a ``sympy.Add`` (``int + Idx``) as their
    # index — only plain slices / integers / Idx.
    T_ib = sp.IndexedBase(T.name)

    # Outer loop index: k ∈ [0, M). Each k corresponds to state cell
    # ``off + k + 1`` in ``T`` (local cell position 1..M in the
    # pipe, cell 0 is BC-pinned and lives outside this LoopOde).
    k = Idx(f'_k_{pipe_name}', M)

    # Head/tail kt1 stencil — same shape at both boundaries.
    # At k==0 (head):   T[off+0], T[off+1]
    # At k==M-1 (tail): T[off+M-1], T[off+M]
    # Both fit the single ``kt1_ode(T[off+k], T[off+k+1], ...)``
    # form as long as we gate it with a (head + tail) mask below.
    kt1_expr = kt1_ode(T_ib[off + k], T_ib[off + k + 1],
                        m, lam, rho, Cp, S, Tamb, dx)
    # Interior kt2 stencil at k ∈ [1, M-2]:
    #   T[off+k-1], T[off+k], T[off+k+1], T[off+k+2]
    #
    # At the edge cells (k==0 and k==M-1) the kt2 expression is
    # masked out, but numba still evaluates every arg before the
    # mask multiplication — so the stencil's out-of-pipe neighbour
    # reads need to land on *valid* flat-Var cells, not past the
    # tail of ``Tsp_all``.
    #
    # Specifically:
    #   - At k == 0 the first arg ``T[off - 1]`` dips into the
    #     previous pipe's segment (or negatively wraps to the
    #     final state cell of the whole flat Var for the first
    #     pipe). Numpy happily returns a value — **but that read
    #     is wasted**, we multiply it by zero via the mask. The
    #     read itself doesn't crash.
    #   - At k == M-1 the last arg ``T[off + M + 1]`` can be
    #     ``total_len`` for the *last* pipe — actually out of
    #     bounds and raising ``IndexError`` on numpy.
    #
    # Fix: clamp the unsafe indices to in-range values via
    # ``KroneckerDelta``-based offset corrections. At k=0 we
    # forward the first arg from ``off-1`` to ``off+1``; at
    # k=M-1 we rewind the last arg from ``off+M+1`` to ``off+M-1``.
    # Both corrected positions are already valid in-pipe cells
    # and the masked multiplication still zeros the whole kt2
    # contribution at the boundaries.
    first_arg_idx = off + k - 1 + 2 * KroneckerDelta(k, 0)
    last_arg_idx = off + k + 2 - 2 * KroneckerDelta(k, M - 1)
    kt2_expr = kt2_ode(T_ib[first_arg_idx], T_ib[off + k],
                        T_ib[off + k + 1], T_ib[last_arg_idx],
                        m, lam, rho, Cp, S, Tamb, theta_sym, dx)
    mask_edge = KroneckerDelta(k, 0) + KroneckerDelta(k, M - 1)
    mask_interior = 1 - mask_edge
    body = mask_edge * kt1_expr + mask_interior * kt2_expr

    artifact['T' + pipe_name + '_loopode'] = LoopOde(
        f'heat_pipe_kt2_loop_T{pipe_name}',
        outer_index=k,
        body=body,
        diff_var=T[off + 1:off + M + 1],
        model=model,
    )
    return artifact
