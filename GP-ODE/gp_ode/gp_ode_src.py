import numpy as np

#
# ToDo - Add F pars maps
class LatentVelocity:
    def __init__(self, F, latentGPs):
        self.dim = len(latentGPs) # Expects latentGPs as a list
        self.latentGPs = latentGPs
        self.F = F
        
    # Calls the simulate method of the pth
    # latent Gaussian process
    def sim_latent(self, p, tp, set=False):
        r = self.latentGPs[p].sim(tp)
        if set:
            self.set_latent(p, tp, r)

    # Set the conditioned points of
    # the pth latent Gaussian process
    def set_latent(self, p, tp, xp):
        self.latentGPs[p].interp_fit(tp, xp)

    # Potential to add an additional sensitivity
    # dependency structure to the latent GPs
    def dXdt(self,X,t=0,indep=True):
        eta = np.array([gp.interp_evalf(t) for gp in self.latentGPs])
        return self.F(X,t) + eta


class MHLatentVelocity:
    def __init__(self, LV, P):
        self.LV = LV
        self.P = P
    
        
