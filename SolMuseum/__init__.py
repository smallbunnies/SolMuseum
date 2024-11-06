# Pls do not import any symbolic function or module here.
# If doing so, Solverz cannot import num_api module independently from SolMuseum because the symbolic functions and
# modules have to be initialized in this __init__.py file where the symbols are imported from Solverz.
# This causes the cirluar ImportError.
# This is ad-hoc. We are working on this for a more elegant solution.
