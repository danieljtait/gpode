import numpy as np


##
# Carries out the matrix inversion
#
#       res = C^{-1} x
#
# using the cholesky factor L of C
def _back_sub(L, x):
    return np.linalg.solve(L.T, np.linalg.solve(L, x))
