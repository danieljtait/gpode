import numpy as np
from collections import namedtuple

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

        if mdate:
            N = self.full_times.size
        else:
            N = self.data_times.size
            
        self.dim = Dimensions(self.model_mats.shape[0],
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


        # logical indicator if we are fitting the version of the model with missing data
        self.missing_data_bool = mdata

        """
        Handles the initalisation of the kernel functions for the trajectory
        and gradient expert.

        see: Initalisation Utility Functions

        """
        _store_gpdx_covs(self)              # ToDo - combine these as one function 
        _store_gpxdx_prior_inv_covs(self):  #
        _store_gpdx_mean_transforms(self)   #
        _store_gp_g_prior_inv_covs(self, g_gp_pars)

        # ready to initalise all of the variational distributions
        self._init_X_var_dist()
        self._init_G_var_dist()


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

    gammas = obj._gammas

    In = np.diag(np.ones(mobj.dim.N))

    for k in range(mobj.dim.K):

        kern = mobj._x_kernels[k]

        Cxx = kern.cov(0, 0, tt, tt)
        Lxx = np.linalg.cholesky(Cxx)

        Cxdx = kern.cov(0, 1, tt, tt)
        Cdxdx = kern.cov(1, 1, tt, tt)

        Cdxdx_x = Cdxdx - np.dot(Cxdx.T, linalg_utils._back_sub(Lxx, Cxdx))
        S = Cdxdx_x + gammas[k]**2*I
        S_chol = linalg_utils.cholesky(S)

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
            linalg_utils._back_sub(Lxx, In))

        obj._dX_prior_inv_covs.append(
            linalg_utils._back_sub(Ldxdx, In))


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
    for r, par in enumerate(gp_g_pars):

        # Base covariance for the latent gp r - depends on the inv length scale
        cov0 = np.exp(-par[1]*(_S.ravel()-_T.ravel())**2).reshape(_T.shape)

        obj._G_prior_inv_covs0.append(
            np.linalg.inv(cov0))
