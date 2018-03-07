import numpy as np

from scipy.linalg import block_diag
from gpode.linlatentforcemodels import (linalg_util,
                                        matrixrv_util)
from collections import namedtuple
from gpode.kernels import GradientMultioutputKernel


Dimensions = namedtuple("Dimensions",
                        "R K N")

class varmlfm_adapgrad:
    def __init__(self,
                 model_mats,
                 data_times,
                 data_Y,
                 sigmas,
                 gammas,
                 x_gp_pars,
                 g_gp_pars,
                 mdata=False):

        ##
        # To Do - allow more general kernels for the
        #         latent trajectory
        _x_kernel_type = "sqexp"

        # Attach the data
        self.data_times = data_times
        self.data_Y = data_Y


        # model matrices
        # ToDo - Add in any linking functions for structured matrices and how
        #        they relate to the model parameters
        self.model_mats = np.asarray(model_mats)
        assert(self.model_mats.shape[1] == self.model_mats.shape[2])  # squareness check

        if mdata:
            N = self.full_times.size
        else:
            N = self.data_times.size
            
        self.dim = Dimensions(self.model_mats.shape[0] - 1,
                              self.model_mats.shape[1],
                              N)

        # Arguable move this to seperate handing function
        # For each out dimension we attach a GradientMultioutputKernel
        if _x_kernel_type == "sqexp":
            _x_kernels = []
            for kp in x_gp_pars:
                _x_kernels.append(GradientMultioutputKernel.SquareExponKernel(kp))
            self._x_kernels = _x_kernels


        # Pseudo-noise variable governing the contribution from the
        # gradient expert
        self._gammas = gammas

        # noise of the data error component. Can be optimised within the
        # variational framework so store the parameters governing its dist.
        #
        # ToDo - allow this parameter to be freely optimized, use precision parameterisation
        self._sigmas = sigmas
        self._sigmas_inv_alphas = 100*np.ones(self.dim.K)
        self._sigmas_inv_betas = 0.01*np.ones(self.dim.K)

        # logical indicator if we are fitting the version of the model with missing data
        self.missing_data_bool = mdata

        """
        Handles the initalisation of the kernel functions for the trajectory
        and gradient expert.

        see: Initalisation Utility Functions

        """
        _store_gpdx_covs(self)              # ToDo - combine these as one function 
        _store_gpxdx_prior_inv_covs(self)   #
        _store_gpdx_mean_transforms(self)   #
        _store_gp_g_prior_inv_covs(self, g_gp_pars)

        # ready to initalise all of the variational distributions
        self._init_X_var_dist()
        self._init_G_var_dist()


    def _init_X_var_dist(self, scale=0.1):
        self._X_means = [np.zeros(self.dim.N)
                         for k in range(self.dim.K)]

        self._X_covars = {}

        for i in range(self.dim.K):
            for j in range(i+1):
                if i == j:
                    _C = np.diag(scale*np.ones(self.dim.N))
                else:
                    _C = np.zeros((self.dim.N, self.dim.N))

                self._X_covars[(i, j)] = _C
                self._X_covars[(j, i)] = _C


    def _init_G_var_dist(self):
        self._G_means = [np.zeros(self.dim.N)
                         for r in range(self.dim.R)]

        self._G_covars = {}
        for s in range(self.dim.R):
            for t in range(s+1):
                if s==t:
                    _C = np.diag(np.ones(self.dim.N))
                else:
                    _C = np.zeros((self.dim.N, self.dim.N))

                self._G_covars[(s, t)] = _C
                self._G_covars[(t, s)] = _C


    def _update_x_var_dist(self):
        # Contribution from the prior
        x_prior_mean = np.zeros(self.dim.N*self.dim.K)
        x_prior_inv_cov = block_diag(
            *(p_ic for p_ic in self._X_prior_inv_covs))

        # Contribution from the data

        # modify to use var par alphas, betas
        E_noise_prec = self._sigmas_inv_alphas/self._sigmas_inv_betas

        if self.missing_data_bool:
            x_data_mean = np.zeros((self.dim.N, self.dim.K))
            for ind, y in zip(self.data_inds, self.data_Y):
                x_data_mean[ind, ] = y
            x_data_mean = x_data_mean.T.ravel()

            sinv_diags = []
            for tau in E_noise_prec:
                i_diag = np.zeros(self.dim.N)
                i_diag[self.data_inds] = tau
                tau_diags.append(i_diag)
            x_data_inv_cov = np.concatenate(tau_diags)

        else:
            x_data_mean = self.data_Y.T.ravel()
            #x_data_inv_cov = np.diag(
            #    np.concatenate([(1/s**2)*np.ones(self.dim.N) for s in self._sigmas])
            #    )
            x_data_inv_cov = np.diag(
                np.concatenate([tau*np.ones(self.dim.N) for tau in E_noise_prec])
                )

        means = [x_prior_mean, x_data_mean]
        inv_covs = [x_prior_inv_cov, x_data_inv_cov]

        # Contribution from the model
        for i in range(self.dim.K):

            Sigma_Inv_i = self._dX_prior_inv_covs[i]
            Pi = self._dX_cond_mean_transform[i]

            m, ic = _parse_component_i_for_x(i, self._G_means, self._G_covars,
                                             Sigma_Inv_i, Pi, self.model_mats,
                                             self.dim.K, self.dim.R, self.dim.N)
            means.append(m)
            inv_covs.append(ic)

        mean, cov = matrixrv_util._prod_norm_pars(means, inv_covs)

        mean = mean.reshape(self.dim.K, self.dim.N)
        for i, mu in enumerate(mean):
            self._X_means[i] = mu

        N = self.dim.N
        for i in range(self.dim.K):
            X_cov_ii = cov[i*N:(i+1)*N, i*N:(i+1)*N]
            self._X_covars[(i, i)] = X_cov_ii
            for j in range(i):
                X_cov_ij = cov[i*N:(i+1)*N, j*N:(j+1)*N]
                self._X_covars[(i, j)] = X_cov_ij
                self._X_covars[(j, i)] = X_cov_ij.T
                

    def _update_g_var_dist(self):
        # Contribution from the prior
        g_prior_mean = np.zeros(self.dim.N*self.dim.R)
        g_prior_inv_cov = block_diag(*(p_ic for p_ic in self._G_prior_inv_covs0))

        means = [g_prior_mean]
        inv_covs = [g_prior_inv_cov]

        for i in range(self.dim.K):
            Sigma_Inv_i = self._dX_prior_inv_covs[i]

            Pi = self._dX_cond_mean_transform[i]
            
            m, ic = _parse_component_i_for_g(i, self._X_means, self._X_covars,
                                             Sigma_Inv_i, Pi, self.model_mats,
                                             self.dim.K, self.dim.R, self.dim.N)
            means.append(m)
            inv_covs.append(ic)
        
        mean, cov = matrixrv_util._prod_norm_pars(means, inv_covs)
        mean = mean.reshape(self.dim.R, self.dim.N)
        
        for r, mu in enumerate(mean):
            self._G_means[r] = mu

        N = self.dim.N
        for s in range(self.dim.R):
            G_cov_ss = cov[s*N:(s+1)*N, s*N:(s+1)*N]
            self._G_covars[(s, s)] = G_cov_ss
            for t in range(s):
                G_cov_st = cov[s*N:(s+1)*N, t*N:(t+1)*N]
                self._G_covars[(s, t)] = G_cov_st
                self._G_covars[(t, s)] = G_cov_st.T



"""
Initalisation Utility Functions

Utility functions to handle initalisation of the varmlfm_adapgrad class
"""
def _store_gpdx_covs(mobj):

    mobj.Lxx = []     # Cholesky factors of the covariance of the trajectories
    mobj.Cxdx = []    # cross-covariance between a trajector and its gradient
    mobj.S_chol = []  # Cholesky decomposition of S = Cdxdx|x + gamma[k]*I

    if mobj.missing_data_bool:
        tt = mobj.full_times
    else:
        tt = mobj.data_times

    gammas = mobj._gammas

    In = np.diag(np.ones(mobj.dim.N))

    for k in range(mobj.dim.K):

        kern = mobj._x_kernels[k]

        Cxx = kern.cov(0, 0, tt, tt)
        Lxx = np.linalg.cholesky(Cxx)

        Cxdx = kern.cov(0, 1, tt, tt)
        Cdxdx = kern.cov(1, 1, tt, tt)

        Cdxdx_x = Cdxdx - np.dot(Cxdx.T, linalg_util._back_sub(Lxx, Cxdx))
        S = Cdxdx_x + gammas[k]**2*In
        S_chol = linalg_util.cholesky(S)

        mobj.Lxx.append(Lxx)
        mobj.Cxdx.append(Cxdx)
        mobj.S_chol.append(S_chol)


###
# will need to store the inverse covariance matrices for variational
# updating
def _store_gpxdx_prior_inv_covs(obj):

    In = np.diag(np.ones(obj.dim.N))

    obj._X_prior_inv_covs = []
    obj._dX_prior_inv_covs = []

    for Ldxdx, Lxx in zip(obj.S_chol, obj.Lxx):

        obj._X_prior_inv_covs.append(
            linalg_util._back_sub(Lxx, In))

        obj._dX_prior_inv_covs.append(
            linalg_util._back_sub(Ldxdx, In))


#################################################################
#                                                               #
# Conditional mean of the gradient exper is given by the linear #
# transformation
#
#     E[dx | x ] = Cdxx Cxx^{-1} x
#
# we store the N x N matrix Cdxx Cxx^{-1}
#
#################################################################
def _store_gpdx_mean_transforms(obj):
    obj._dX_cond_mean_transform = []
    for Cxdx, Cinv in zip(obj.Cxdx, obj._X_prior_inv_covs):
        obj._dX_cond_mean_transform.append(
            np.dot(Cxdx.T, Cinv)
            )

#################################################################
#
#################################################################
def _store_gp_g_prior_inv_covs(obj, g_gp_pars):

    if obj.missing_data_bool:
        tt = obj.full_times
    else:
        tt = obj.data_times

    # base inv cov - cov will be scale multiple of this
    obj._G_prior_inv_covs0 = []

    _T, _S = np.meshgrid(tt, tt)
    for par in g_gp_pars:

        # Base covariance for the latent gp r - depends on the inv length scale
        cov0 = np.exp(-par[1]*(_S.ravel()-_T.ravel())**2).reshape(_T.shape)

        obj._G_prior_inv_covs0.append(
            np.linalg.inv(cov0))

"""
Model Fitting Functions
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

##########################################
#
#     Ws_i[j] = sum_s A[s, i, j] o g_s
#
##########################################
def _get_Wi_mean(EG, i, A):
    EWs_i = [sum(A[s+1, i, j]*Egs + A[0, i, j]
                 for s, Egs in enumerate(EG))
             for j in range(A.shape[1])]
    return EWs_i


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
        [matrixrv_util._E_diagx_M_y(E_Vi[s], E_Vi[0], Cov_Vi[(s, 0)], Sigma_Inv)
         for s in range(1, R+1)]
        )
    E_ViT_Sinv_Pixi = np.concatenate(
        [matrixrv_util._E_diagx_M_y(E_vs, np.dot(Pi, EX[i]), cvsmi, Sigma_Inv)
         for E_vs, cvsmi in zip(E_Vi[1:], Cov_Vi_Mi)]
        )

    try:
        mean = np.linalg.solve(inv_covar, E_ViT_Sinv_Pixi - E_ViT_Sinv_v0 )
    except:
        pinv = np.linalg.pinv(inv_covar)
        mean = np.dot(pinv, E_ViT_Sinv_Pixi - E_ViT_Sinv_v0 )
    
    return mean, inv_covar
