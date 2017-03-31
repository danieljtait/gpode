import numpy as np
import scipy.stats
from scipy.special import erf

## 
# Fitting routines for the latent variable
# force model in Alvarez et. al 

rootPi = np.sqrt(np.pi)

def h(s,t,
      q,p,r,
      rho, nu_rq, D):

    expr0 = np.exp(nu_rq*nu_rq - D[q]*s)/(D[p] + D[q])

    expr1 = np.exp(D[q]*t)
    expr1 *= erf( rho[r]*(s-t) - nu_rq) + erf( rho[r]*t + nu_rq )

    expr2 = np.exp(-D[q]*t)
    expr2 *= erf( rho[r]*s - nu_rq) + erf(nu_rq)

    return expr0*(expr1 - expr2)

def ky(p,q,s,t,
       rho, nu_rq, D,
       S):
    s = np.array(s)
    print s
    k = np.zeros(s.size)
    for r in range(D.size):
        k += S[r,p]*S[r,q]*rootPi*rho[r]*(h(s,t,q,p) + h(t,s,p,q))
    return 0.5*k    

def kyf(p, q, s, t,
        rho, nu_rq, D, S):

    expr0 = 0.5*S[q,p]*rootPi/rho[p]*np.exp(nu_rq*nu_rq)

    expr1 = np.exp(-D[p]*(s-t))

    expr2 = erf( (t-s)/rho[r] - nu_rq ) + erf( t/rho[r] + nu_rq )

    return expr0*expr1*expr2




class GP_LVFM:
    def __init__(self, D, B, S, kernel_par):
        self.D = D
        self.B = np.array(B)
        self.S = S
        self.dim = self.B.size

        # Parameterised by the square
        # exponential kernel

        # - not consistent with other implementations in same package
        self.kernel_par = kernel_par
        self.rho = [1./np.sqrt(tau) for tau in kernel_par]

    def h(self,s, t, q, p, r):
        nu_rq = 0.5*self.D[q]/self.rho[r]
        return h(s, t, q, p, r,
                 self.rho, nu_rq, self.D)


    def ky(self, s, t, p, q):
        s = np.array(s)
        k = np.zeros(s.size)
        for r in range(self.dim):
            k += self.S[r,p]*self.S[r,q]*rootPi*self.rho[r]*(self.h(t,s,q,p,r) + self.h(s,t,p,q,r))
        return 0.5*k


  

def output_mean(t, y0, lv):
    return y0*np.exp(-lv.D*t) + lv.B/lv.D

def make_output_cov(Ts, lv):
    dim = len(Ts)

    n0 = Ts[0].size
    S00, T00 = np.meshgrid(Ts[0], Ts[0])
    C = []

    for p in range(dim):
        Np = Ts[p].size
        Sp, Tp = np.meshgrid(Ts[p], Ts[p])
        # Scale by 1/2 
        Cp = 0.5*lv.ky(Sp.T.ravel(), Tp.T.ravel(), p, p).reshape(Np, Np)
        for q in range(dim-1-p):
            Nq = Ts[q+1].size
            Sp, Tq = np.meshgrid(Ts[p], Ts[q])
            Cpq = lv.ky(Sp.T.ravel(), Tq.T.ravel(), p, q+1).reshape(Np, Np)
            Cp = np.column_stack((Cp, Cpq))

        if p == 0 :
            C = Cp.copy()
        else:
            O = np.zeros((Np, C.shape[1] - Cp.shape[1]))
            Cp = np.column_stack((O, Cp))
            C = np.row_stack((C, Cp))

    return C + C.T
