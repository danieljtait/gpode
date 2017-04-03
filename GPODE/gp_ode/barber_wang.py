import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as deriv


def ksqexp(s, t, par, n=0):
    if n == 0:
        return par[0]*np.exp(-0.5*par[1]*(s-t)**2)
    elif n == 1:
        kd = deriv(lambda t_: ksqexp(s, t_, par, 0), x0=t, dx=1e-6)
        return kd
    elif n == 2:
        kdk = deriv(lambda s_: ksqexp(s_, t, par, 1), x0=s, dx=1e-6)
        return kdk

# The Barber-Wang Gaussian Process ODE model

# for compatibility with arbitrary dimensions require a tuple of kernels
class gp_ode_bw:
    def __init__(self, F, kernels, ktypes='sqexp', kernels_par=None, ode_class=None):
        self.kernels = kernels
        self.kernels_par = kernels_par
        self.dim = len(kernels)

        if isinstance(ktypes, basestring):
            self.kernel_types = []
            for d in range(self.dim):
                self.kernel_types.append(ktypes)
        else:
            self.kernel_types = ktypes

        # Cetain aspects need to be handled differently if ode
        # is of the form dx/dt = F(x,t) + G(t), for some Gaussian Process G(t)
        self.ode_class = ode_class
        self.F = F

        self.eval_ts = None
        self.eval_xs = None

        # small correction for singular matrices
        self.diag_corr = 1e-3

    #################################################
    # As noted in Macdonald et al 2015 the model    #
    # is driven by representitve latent states that #
    # determine the derivative                      #
    #################################################
    def set_latent_states(self, tt, xx, k='All'):
        self.eval_ts = tt
        xx = np.array(xx)

        self.rep_latent_states = xx
        self.rep_latent_states_deriv = np.array([self.F(x,t) for x,t in zip(self.rep_latent_states, tt)])

    ##
    #  the kth component of the latent state
    # given the representitive states
    def interp_latent_states_evalf(self, tev, k):
        Np = self.eval_ts.size

        
        S, T = np.meshgrid(self.eval_ts, self.eval_ts)
        
        C00 = self.kernels[k](S.ravel(), T.ravel(), self.kernels_par[k]).reshape(Np, Np)
        C01 = self.kernels[k](S.ravel(), T.ravel(), self.kernels_par[k],1).reshape(Np, Np)
        C11 = self.kernels[k](S.ravel(), T.ravel(), self.kernels_par[k],2).reshape(Np, Np)
                

        try:
            self.dLd = np.linalg.cholesky(C11)
        except:
            print "Singular covariance matrix... adding small multiple of identity."
            self.dLd = np.linalg.cholesky(C11 + np.diag(self.diag_corr*np.ones(Np)))

        # cov of
        kk = np.array([self.kernels[k](tev, s, self.kernels_par[k], 1) for s in self.eval_ts])
            
        a = self.rep_latent_states_deriv[:,k]

        s1 = np.linalg.solve(self.dLd, a)
        s2 = np.linalg.solve(self.dLd.T, s1)

        mc = np.dot(kk, s2)

        return mc


class gp_ode_bw_lf:
    def __init__(self, F, kernels, ktypes='sqexp', kernels_par=None):
        self.kernels = kernels
        self.kernels_par = kernels_par
        self.dim = len(kernels)

        if isinstance(ktypes, basestring):
            self.kernel_types = []
            for d in range(self.dim):
                self.kernel_types.append(ktypes)
        else:
            self.kernel_types = ktypes

        self.eval_ts = None
        self.eval_xs = None

        self.F = F

        # Small correction term for singular cov matrices
        self.diag_corr = 1e-3

    # Sets latent states including latent Gaussian force
    def set_latent_states(self, tt, xx, g_xx, k='All'):
        self.eval_ts = tt
        xx = np.array(xx)

        self.rep_latent_states = xx
        self.rep_latent_gp_forces = g_xx
        self.rep_latent_states_deriv = np.array([self.F(x,t,g_xx) for x, t in zip(self.rep_latent_states, tt)])
            
    

##
# Full model, includes
#
# [ ] model prior distribution
# [ ] kernel parameter prior distributions
# [ ] full joint distribution
#
#
# Conditionals:
#
# - 
# 
#class gp_ode_model:




class gp_ode_bw_gibbs_sampler:
    def __init__(self, gp, ):
        self.gp = gp # Gaussian Process object

