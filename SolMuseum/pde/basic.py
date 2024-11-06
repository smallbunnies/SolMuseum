from Solverz.sym_algebra.functions import MulVarFunc


class SolPde(MulVarFunc):
    def _numpycode(self, printer, **kwargs):
        return (f'SolMF.pde.{self.__class__.__name__}' + r'(' +
                ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')')
