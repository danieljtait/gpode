import numpy as np
import scipy.stats


def ksqexp(s, t, l=1.):
    return np.exp(-0.5*l*(s-t)**2)


class GaussianProcess:
    def __init__(self,kernel,ktype='sqexp',kernel_par=None, mean_func=None):
        self.kernel=kernel
        self.kernel_type = ktype
        self.kernel_par=kernel_par

        if mean_func == None:
            self.mean_func = lambda t: 0.

        self.eval_t = None # Observation points
        self.eval_x = None # fitted data

        self.diag_corr = 1e-3 # Small correction term in the case of singular covar matrix

    # Evaluate the log probability of the Gaussian process
    def logpdf(self, x, eval_t = None):
        eval_t = np.array(eval_t)
        if eval_t == None:
            eval_t = self.eval_t
        S, T = np.meshgrid(eval_t,eval_t)
        C = self.kernel(S.ravel(), T.ravel(), self.kernel_par).reshape(eval_t.size, eval_t.size)
        m = [self.mean_func(t) for t in eval_t]
        try:
            return scipy.stats.multivariate_normal.logpdf(x, mean=m, cov=C)
        except:
            C+= np.diag(self.diag_corr*np.ones(eval_t.size))
            return scipy.stats.multivariate_normal.logpdf(x, mean=m, cov=C)
        
    # Move all the fitting routines into a seperate function
    def fit_loglik(self, Y, m, Sigma):
        return scipy.stats.multivariate_normal.logpdf(Y, mean=m, cov=Sigma)

    def fit_objFunc(self, par, Y, tp):
        S, T = np.meshgrid(tp, tp)
        # Construct the covariance function for given parameter
        C = self.kernel(S.ravel(), T.ravel(), par).reshape(tp.size, tp.size)
        m = [self.mean_func(t) for t in tp]
        return self.fit_loglik(Y, m, C)

    def fit(self, Y, tp, add_err=False):
        print self.fit_objFunc(1., Y, tp)
        # Do something
        return 0.

    def sim(self, tp):
        S, T = np.meshgrid(tp, tp)
        C = self.kernel(S.ravel(), T.ravel(), self.kernel_par).reshape(tp.size, tp.size)

        m = [self.mean_func(t) for t in tp]

        try:
            return scipy.stats.multivariate_normal.rvs(mean=m, cov=C)
        except:
            C += np.diag(self.diag_corr*np.ones(tp.size))
            return scipy.stats.multivariate_normal.rvs(mean=m, cov=C)

    def interp_fit(self, t, Y):
        self.eval_t = np.array(t)
        self.eval_t_mean = np.array([self.mean_func(t) for t in self.eval_t])
        
        self.eval_x = np.array(Y)
        self.Np = self.eval_t.size

        S, T = np.meshgrid(self.eval_t,self.eval_t)
        C = self.kernel(S.ravel(), T.ravel(), self.kernel_par).reshape(self.Np, self.Np)

        # Store the lower triangle of the Cholesky decomposition of C
        try:
            self.L = np.linalg.cholesky(C)
        except:
            print "Singular covariance matrix... adding small multiple of identity matrix."
            self.L = np.linalg.cholesky(C + np.diag(self.diag_corr*np.ones(self.Np)))


    def interp_evalf(self, tev, wVar=False):
        # covariance between the evaluation point and the interpolation points
        kk = np.array([self.kernel(tev, s, self.kernel_par) for s in self.eval_t])

        # mean at evaluation point
        m1 = self.mean_func(tev)

        # mean at interpolation knots
        m2 = self.eval_t_mean

        # perform the least squares regression problem with the Cholesky
        # decomposition of the covariance matrix
        s1 = np.linalg.solve(self.L, self.eval_x - self.eval_t_mean)
        s2 = np.linalg.solve(self.L.T, s1)

        # The conditional mean given the interpolation points
        mc = m1 + np.dot(kk, s2)

        # Just return the point estimate
        if not wVar:
            return mc
        # Otherwise also return the variance at that point
        else:
            C = np.dot(self.L, self.L.T)
            Cinv = np.linalg.inv(C)
            v = self.kernel(tev,tev, self.kernel_par) - np.dot(kk, np.dot(Cinv, kk))
            return mc, v



