import numpy as np

##
#  p(x) ∝ Π N(x, means[k] | inv_covs[k])
def _prod_norm_pars(means, inv_covs):
    m1 = means[0]
    C1inv = inv_covs[0]

    if len(means) == 1:
        return m1, np.linalg.inv(C1inv)

    else:

        for m2, C2inv in zip(means[1:], inv_covs[1:]):
            Cnew_inv = C1inv + C2inv
            mnew = np.linalg.solve(Cnew_inv,
                                   np.dot(C1inv, m1) + np.dot(C2inv, m2))
            m1 = mnew
            C1inv = Cnew_inv

        return mnew, np.linalg.inv(Cnew_inv)


######################################################
#                                                    #
# E[diag(x) M y ] = np.sum(Exyt * M, axis=1)         #
#                                                    #
######################################################
def _E_diagx_M_y(E_x, E_y, Cov_xy, M):
    ExyT = Cov_xy + np.outer(E_x, E_y)
    return np.sum(ExyT * M, axis=1)
