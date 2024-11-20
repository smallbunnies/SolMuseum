from Solverz.sym_algebra.functions import MulVarFunc
from sympy import Integer


class SolPde(MulVarFunc):
    def _numpycode(self, printer, **kwargs):
        return (f'SolMF.pde.{self.__class__.__name__}' + r'(' +
                ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')')


class minmod(SolPde):
    arglength = 3

    def _eval_derivative(self, s):
        return switch_minmod(*[arg.diff(s) for arg in self.args],
                             minmod_flag(*self.args))


class minmod_flag(SolPde):
    """
    Different from `minmod`, minmod function outputs the position of args instead of the values of args.
    """
    arglength = 3

    def _eval_derivative(self, s):
        return Integer(0)


class switch_minmod(SolPde):
    arglength = 4

    def _eval_derivative(self, s):
        return switch_minmod(*[arg.diff(s) for arg in self.args[0:len(self.args) - 1]], self.args[-1])
