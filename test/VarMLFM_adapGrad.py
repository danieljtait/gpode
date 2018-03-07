import numpy as np
import linalg_utils
import matrixrv_utils
from scipy.linalg import block_diag
from gpode.kernels import GradientMultioutputKernel


class VarMLFM_adapGrad:

    def __init__(self,
                 ModelMatrices,
                 data_times,
                 data_Y,
                 sigmas,
                 gammas,
                 x_gp_pars,
                 mdata=False):

        _x_kernel_type = "sqexp"

        # Attach the data
        self.data_times = data_times
        self.data_Y = data_Y

        # Model matrices
        self.A = np.asarray(ModelMatrices)

        self._R = self.A.shape[0] - 1       # R is the number of latent forces
                                            # there is an additional constant matrix
                                            # A[0]
        self._K = self.A.shape[1]           # Observation dimension

        if mdata:                           # Number of observations
            self._N = self.full_times.size
        else:
            self._N = data_times.size       
        
        assert(self._K == self.A.shape[2])  # Squareness check

        ###
        # For each component k we attach a GradientMultioutputKernel
        if _x_kernel_type == "sqexp":
            _x_kernels = []
            for kp in x_gp_pars:
                _x_kernels.append(GradientMultioutputKernel.SquareExponKernel(kp))

            self._x_kernels = _x_kernels

        # pseudo-noise variable governing the contribution from the
        # gradient exper
        self._gammas = gammas

        # noise variables
        self._sigmas = sigmas 

        
        ##
        # Attach the gradient kernel covariance objects to the class
        # - the characteristic length scale parameters (those not possessing
        #   tractable marginals, will typically not be changed during a call to
        #   fit
        self.missing_data = mdata
        _store_gpdx_covs(self)
        _In = np.diag(np.ones(self._N))

        self._X_prior_inv_covs = []
        self._dX_prior_inv_covs = []

        for Ldxdx, Lxx in zip(self.S_chol, self.Lxx):

            self._X_prior_inv_covs.append(
                linalg_utils._back_sub(Lxx, _In))

            self._dX_prior_inv_covs.append(
                linalg_utils._back_sub(Ldxdx, _In))

        ##################
        # Cond mean of the gradient expert corresponds to
        #           
        #     E[dx | x] = Cdxx Cxx^{-1} x
        #
        self._dX_cond_mean_transform = []
        for Cxdx, Cinv in zip(self.Cxdx, self._X_prior_inv_covs):
            self._dX_cond_mean_transform.append(
                np.dot(Cxdx.T, Cinv)
                )

        self._G_prior_inv_covs = []

        if mdata:
            tt = self.full_times
        else:
            tt = self.data_times

        for r in range(self._R):
            cov = np.array([[np.exp(-(s-t)**2) for t in tt]
                            for s in tt])
            self._G_prior_inv_covs.append(np.linalg.inv(cov))

        self._init_X_var_dist()
        self._init_G_var_dist()

    def _init_X_var_dist(self, scale=0.1):
        self._X_means = [np.zeros(self._N)
                         for k in range(self._K)]
        self._X_covars = {}

        for i in range(self._K):
            for j in range(i+1):
                if i == j:
                    _C = np.diag(scale*np.ones(self._N))
                else:
                    _C = np.zeros((self._N, self._N))
                self._X_covars[(i, j)] = _C
                self._X_covars[(j, i)] = _C

    def _init_G_var_dist(self):
        self._G_means = [np.zeros(self._N)
                         for r in range(self._R)]

        self._G_covars = {}
        for s in range(self._R):
            for t in range(self._R):
                if s == t:
                    _C = np.diag(np.ones(self._N))
                else:
                    _C = np.zeros((self._N, self._N))
                self._G_covars[(s, t)] = _C
                self._G_covars[(t, s)] = _C


    def _update_X_var_dist(self):
        # Contribution from the prior 
        x_prior_mean = np.zeros(self._N*self._K)
        x_prior_inv_cov = block_diag(*(p_ic for p_ic in self._X_prior_inv_covs))

        # Contribution from the data
        if self.missing_data:
            x_data_mean = np.zeros((self._N, self._K))
            for ind, y in zip(self.data_inds, self.data_Y):
                x_data_mean[ind, ] = y
            x_data_mean = x_data_mean.T.ravel()

            sinv_diags = []
            for s in self._sigmas:
                i_diag = np.zeros(self._N)
                i_diag[self.data_inds] = 1/s**2
                sinv_diags.append(i_diag)
            x_data_inv_cov = np.diag(np.concatenate(sinv_diags))
            
        else:
            x_data_mean = self.data_Y.T.ravel()
            x_data_inv_cov = np.diag(
                np.concatenate([1/(s**2)*np.ones(self._N) for s in self._sigmas])
                )

        means = [x_prior_mean, x_data_mean]
        inv_covs = [x_prior_inv_cov, x_data_inv_cov]

        # Contribution from the model
        for i in range(self._K):

            Sigma_Inv_i = self._dX_prior_inv_covs[i]
            Pi = self._dX_cond_mean_transform[i]
            
            m, ic = _parse_component_i_for_x(i, self._G_means, self._G_covars,
                                             Sigma_Inv_i, Pi, self.A,
                                             self._K, self._R, self._N)
            means.append(m)
            inv_covs.append(ic)

        mean, cov = matrixrv_utils._prod_norm_pars(means, inv_covs)

        mean = mean.reshape(self._K, self._N)
        for i, mu in enumerate(mean):
            self._X_means[i] = mu

        N = self._N
        for i in range(self._K):
            X_cov_ii = cov[i*N:(i+1)*N, i*N:(i+1)*N]
            self._X_covars[(i, i)] = X_cov_ii
            for j in range(i):
                X_cov_ij = cov[i*N:(i+1)*N, j*N:(j+1)*N]
                self._X_covars[(i, j)] = X_cov_ij
                self._X_covars[(j, i)] = X_cov_ij.T

    def _update_G_var_dist(self):
        # Condtribution from the prior
        g_prior_mean = np.zeros(self._N*self._R)
        g_prior_inv_cov = block_diag(*(p_ic for p_ic in self._G_prior_inv_covs))

        means = [g_prior_mean]
        inv_covs = [g_prior_inv_cov]

        # Contribution from the model
        for i in range(self._K):
            Sigma_Inv_i = self._dX_prior_inv_covs[i]
            Pi = self._dX_cond_mean_transform[i]
            
            m, ic = _parse_component_i_for_g(i, self._X_means, self._X_covars,
                                             Sigma_Inv_i, Pi, self.A,
                                             self._K, self._R, self._N)
            means.append(m)
            inv_covs.append(ic)

        mean, cov = matrixrv_utils._prod_norm_pars(means, inv_covs)
        mean = mean.reshape(self._R, self._N)
        
        for r, mu in enumerate(mean):
            self._G_means[r] = mu

        N = self._N
        for s in range(self._R):
            G_cov_ss = cov[s*N:(s+1)*N, s*N:(s+1)*N]
            self._G_covars[(s, s)] = G_cov_ss
            for t in range(s):
                G_cov_st = cov[s*N:(s+1)*N, t*N:(t+1)*N]
                self._G_covars[(s, t)] = G_cov_st
                self._G_covars[(t, s)] = G_cov_st.T


class VarMLFM_adapGrad_missing_data(VarMLFM_adapGrad):
    def __init__(self,
                 ModelMatrices,
                 full_times,
                 data_inds,
                 data_Y,
                 sigmas,
                 gammas,
                 x_gp_pars):

        assert(data_Y.shape[0] == len(data_inds))
        
        self.full_times = full_times
        self.data_inds = np.asarray(data_inds)
        super(VarMLFM_adapGrad_missing_data, self).__init__(ModelMatrices,
                                                            full_times[data_inds],
                                                            data_Y,
                                                            sigmas,
                                                            gammas,
                                                            x_gp_pars,
                                                            mdata=True)


"""
Model Setup Utility Functions
"""
def _store_gpdx_covs(mobj):
    mobj.Lxx = []
    mobj.Cxdx = []
    mobj.S_chol = []

    if mobj.missing_data:
        tt = mobj.full_times
    else:
        tt = mobj.data_times

    gammas = mobj._gammas

    for k in range(mobj._K):

        kern = mobj._x_kernels[k]

        Cxx = kern.cov(0, 0, tt, tt)
        Lxx = np.linalg.cholesky(Cxx)

        Cxdx = kern.cov(0, 1, tt, tt)
        Cdxdx = kern.cov(1, 1, tt, tt)

        Cdxdx_x = Cdxdx - np.dot(Cxdx.T, linalg_utils._back_sub(Lxx, Cxdx))
        I = np.diag(np.ones(Cdxdx_x.shape[0]))
        S = Cdxdx_x + gammas[k]**2*I
        S_chol = np.linalg.cholesky(S)

        mobj.Lxx.append(Lxx)
        mobj.Cxdx.append(Cxdx)
        mobj.S_chol.append(S_chol)


"""
Model Fitting Utility Functions
"""

######################################################
#                                                    #
# Vs_i[r] = sum_j (A[r, i, j] o x_j                  #
#                                                    #
# so E[Vs_i[r]] = sum_j (A[r, i, j] o E[x_j]         #
#                                                    #
######################################################
def _get_Vi_mean(EX, i, A):
    Vs_i = [sum(a[i, j]*Exj for j, Exj in enumerate(EX))
          for a in A]
    return Vs_i

######################################################
#                                                    #
# Ws_i[j] = sum_s A[s, i, j] o g_s 
#                                                    #
######################################################
def _get_Wi_mean(EG, i, A):
    Ws_i = [sum(A[s+1, i, j]*Egs + A[0, i, j]
                for s, Egs in enumerate(EG))
            for j in range(A.shape[1])]
    return Ws_i

######################################################
#                                                    #
# Returns a dict of such that                        #
#                                                    #
#     res_dict[(s, t)] =                             #
#         sum_m^K sum_n^K                            #
#              A[s, i, m] Cov{X_m,X_n}  A[t, j,n]    #
#                                                    #
######################################################
def _get_Vi_cov(CovX_dict, i, A, K, R, N):

    I = np.diag(np.ones(N))

    full_cov = np.row_stack((
        np.column_stack((CovX_dict[(i, j)] for j in range(K)))
        for i in range(K)))

    res_dict = {}
    for s in range(R+1):
        asi = np.column_stack((a*I for a in A[s, i, :]))        
        for t in range(s+1):
            ati = np.column_stack((a*I for a in A[t, i, :]))
            res = np.dot(asi, np.dot(full_cov, ati.T))

            res_dict[(s, t)] = res
            res_dict[(t, s)] = res.T

    return res_dict

###
# Returns a dict such that
#
#     res_dict[(m, n)] =
#         sum_s^R sum_t^R A[s, i, m
def _get_Wi_cov(CovG_dict, i, A, K, R, N):

    I = np.diag(np.ones(N))

    full_cov = np.row_stack((
        np.column_stack((CovG_dict[(s, t)] for t in range(R)))
        for s in range(R)))

    res_dict = {}
    for m in range(K):
        for n in range(K):
            cov = np.zeros((N, N))
            for s in range(R):
                for t in range(R):
                    cov += CovG_dict[(s, t)]*A[s+1, i, m]*A[t+1, i, n]
            res_dict[(m, n)] = cov

    return res_dict


def _parse_component_i_for_x(i, EG, CovG,
                             Sigma_Inv, Pi, A,
                             K, R, N):

    E_Wi = _get_Wi_mean(EG, i, A)
    Cov_Wi = _get_Wi_cov(CovG, i, A, K, R, N)

    inv_covar = np.zeros((N*K, N*K))
    for m in range(K):
        for n in range(K):
            res = (Cov_Wi[(m, n)] + np.outer(E_Wi[m], E_Wi[n])) * Sigma_Inv


            if m == i:
                res -= np.dot(Pi.T, np.dot(Sigma_Inv, np.diag(E_Wi[n])))

                if n == i:
                    res -= np.dot(np.diag(E_Wi[m]), np.dot(Sigma_Inv, Pi))
                    res += np.dot(Pi.T, np.dot(Sigma_Inv, Pi))
            elif n == i:
                res -= np.dot(np.diag(E_Wi[m]), np.dot(Sigma_Inv, Pi))

            inv_covar[m*N:(m+1)*N, n*N:(n+1)*N] = res

    return np.zeros(N*K), inv_covar



###
#
# mi = Pi xi
#
# Vi = [d(v1),...,d(vR)]
#
def _parse_component_i_for_g(i, EX, CovX,
                             Sigma_Inv, Pi, A,
                             K, R, N):

    E_Vi = _get_Vi_mean(EX, i, A)
    Cov_Vi = _get_Vi_cov(CovX, i, A, K, R, N)
    
    Cov_Vi_Xi = [sum(a[i, j]*CovX[(j, i)] for j in range(K))
                  for a in A[1:]]
    Cov_Vi_Mi = [np.dot(cvx, Pi) for cvx in Cov_Vi_Xi]
    
    inv_covar = np.row_stack((
        np.column_stack(
        ((Cov_Vi[(s,t)] + np.outer(E_Vi[s], E_Vi[t]))*Sigma_Inv
         for t in range(1, R+1))
        ) for s in range(1, R+1)))

    # E[Vi^T Sinv v0]
    E_ViT_Sinv_v0 = np.concatenate(
        [matrixrv_utils._E_diagx_M_y(E_Vi[s], E_Vi[0], Cov_Vi[(s, 0)], Sigma_Inv)
         for s in range(1, R+1)]
        )
    E_ViT_Sinv_Pixi = np.concatenate(
        [matrixrv_utils._E_diagx_M_y(E_vs, np.dot(Pi, EX[i]), cvsmi, Sigma_Inv)
         for E_vs, cvsmi in zip(E_Vi[1:], Cov_Vi_Mi)]
        )

    try:
        mean = np.linalg.solve(inv_covar, E_ViT_Sinv_Pixi - E_ViT_Sinv_v0 )
    except:
        pinv = np.linalg.pinv(inv_covar)
        mean = np.dot(pinv, E_ViT_Sinv_Pixi - E_ViT_Sinv_v0 )
    
    return mean, inv_covar
