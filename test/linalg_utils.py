import numpy as np


##
# Matrix inversion of the equation
#
#     (LL.T)^{-1} z = x
# by backwards solving with the cholesky factor L
#
def _back_sub(L, x):
    return np.linalg.solve(L.T, np.linalg.solve(L, x))
