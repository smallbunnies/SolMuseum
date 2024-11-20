# SolMuseum
 This library encapsulates the common simulation models for 
 Solverz-based simulation modelling. It is just required that one inputs the model parameters and then the simulation
 model instances are initialized without having to derive them symbolically and jit-compile the case-specified numerical
 codes, which saves the computation overhead.

All the models that have been implemented can be found in the [docs](https://solmuseum.solverz.org) with mathematical 
derivations.

# Installation

SolMuseum requires ```python>=3.10```, and can be installed locally with

```shell
pip install SolMuseum
```

# Usage Example

For example, we want to perform the finite difference of the heat pipe with the Yao's scheme. The illustrative code
We can just import the `heat_pipe` class from `SolMuseum.pde`, input the parameters and set the method to be 'yao'. 
The finite difference equations are automatically derived and added to the `Model` instance.
snippet is

```python
import numpy as np
from Solverz import Var, Param, Model
from SolMuseum.pde import heat_pipe

# modelling
L = 9250
dx = 370
M = int(L / dx)
Tinitial = np.zeros(M + 1)
lam = 1 / 0.35
Cp = 4182
Ta = -10
D = 1.4
S = np.pi * (D / 2) ** 2
rho = 960
Tamb = -10

for i in range(0, len(Tinitial)):
    fai = np.exp(-lam * i * dx / (Cp * np.abs(2543.5)))
    Tinitial[i] = 90.1724637976347 * fai + Ta * (1 - fai)

m = Model()
m.T = Var('T', Tinitial)
m.m = Param('m', 10)
m.lam = Param('lam', lam)
m.rho = Param('rho', rho)
m.S = Param('S', S)
m.Tamb = Param('Tamb', Tamb)
m.Cp = Param('Cp', Cp)
m.dt = Param('dt', 180)
m.__dict__.update(heat_pipe(m.T,
                            m.m,
                            m.lam,
                            m.rho,
                            m.Cp,
                            m.S,
                            m.Tamb,
                            dx,
                            m.dt,
                            M,
                            '1',
                            method='yao'))

```

For reproducible codes of each model, please refer to the test folders, which can serve as the detailed tutorials.
