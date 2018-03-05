import numpy as np
import scipy.stats
from collections import OrderedDict


class ProbabilityDistribution:
    def __init__(self, pdf=None, rvs=None,
                 logpdf=None, logpdf_deriv=None):
        self.pdf = pdf
        self.rvs = rvs
        self.logpdf = logpdf


####################################
# shape α > 0
# scale θ > 0
#
#   α-1  -x/θ /
#  x    e    /
#           /      α
#          / Γ(α)*θ
####################################
class GammaDistribution:
    def __init__(self, a=1, scale=1):
        self.a = a
        self.scale = scale

    def pdf(self, x):
        return scipy.stats.gamma.pdf(x, a=self.a, scale=self.scale)

    def logpdf(self, x, deriv=False):
        if deriv:
            return -1/self.scale + (self.a-1)/x
        else:
            return scipy.stats.gamma.logpdf(x, a=self.a, scale=self.scale)

    def rvs(self, size=1):
        res = scipy.stats.gamma.rvs(a=self.a, scale=self.scale, size=size)
        if size == 1:
            return float(res)
        else:
            return res


class ProposalDistribution(ProbabilityDistribution):
    def __init__(self,
                 pdf=None, rvs=None,
                 is_symmetric=False):
        super(ProposalDistribution, self).__init__(pdf, rvs)
        self.is_symmetric = is_symmetric


##
# These are all a little weird at the moment in that we are
# attaching the parameters of the prior and proposal the
# parameter cls object
def handle_prior_assignment(cls, tup):
    if tup[0] == "gamma":
        cls.prior_hyperpar = tup[1]
        cls.prior = GammaDistribution(a=cls.prior_hyperpar[0],
                                      scale=cls.prior_hyperpar[1])
    elif tup[0] == "unif":
        cls.prior_hyperpar = tup[1]
        cls.prior = scipy.stats.uniform(loc=tup[1][0],
                                        scale=tup[1][1]-tup[1][0])
    else:
        raise ValueError


def handle_proposal_assignment(cls, tup):
    if tup[0] == "normal rw":
        cls.proposal_hyperpar = tup[1]
        q = ProposalDistribution(
            rvs=lambda xcur: np.random.normal(loc=xcur,
                                              scale=cls.proposal_hyperpar),
            is_symmetric=True)

        cls.proposal = q


###
# Parameter class definitions etc
class Parameter:
    def __init__(self, name, prior=None, proposal=None, value=None):
        self.name = name

        if isinstance(prior, tuple):
            handle_prior_assignment(self, prior)
        else:
            self.prior = prior

        if isinstance(proposal, tuple):
            handle_proposal_assignment(self, proposal)
        else:
            self.proposal = proposal

        self.value = value


class ParameterCollection:
    def __init__(self, parameters, independent=False):
        self.parameters = OrderedDict((p.name, p) for p in parameters)
        self.independent_collection = independent

        if independent:
            def _pdf(xx):
                ps = [p.prior.pdf(x)
                      for p, x in zip(self.parameters.values(), xx)]
                return np.prod(ps)

            def _logpdf(xx, deriv=False):
                lps = [p.prior.logpdf(x, deriv)
                       for p, x in zip(self.parameters.values(), xx)]
                if deriv:
                    return np.array(lps)
                else:
                    return np.sum(lps)

            def _rvs():
                rv = [p.prior.rvs() for p in self.parameters.values()]
                return rv

            self.prior = ProbabilityDistribution(_pdf, _rvs,
                                                 logpdf=_logpdf)

            def _qpdf(xx, xcur):
                qs = [p.proposal.pdf(x, xc)
                      for p, x, xc in zip(self.parameters.values(), xx, xcur)]
                return qs

            # Note at the moment this returns a column vector
            def _qrvs(xcur):
                rv = [p.proposal.rvs(x)
                      for p, x in zip(self.parameters.values(), xcur)]
                return rv

            is_sym = all([p.proposal.is_symmetric
                          for p in self.parameters.values()])

            self.proposal = ProposalDistribution(_qpdf, _qrvs, is_sym)

    def value(self, np_arrayfy=False, arr_shape=None):
        v = [item[1].value for item in self.parameters.items()]

        if np_arrayfy:
            v = np.array(v)
            if arr_shape is not None:
                v.shape = arr_shape

        return v


#################################################
#
# For a propobability distribution
#
#     p(z) ∝ exp(-E(z))
#
# carries out Hamiltonian Monte Carlo Samping from the distribution
#
#     p(z, r) ∝ exp(-H(z, r)), H(z, r) = E(z) + K(r),
#
# with K(r) = sum r^2
#
# ToDo:
#
#    [ ] handle reporting of the rejections for tuning purposes
#
######
class HamiltonianMonteCarlo:
    def __init__(self, E, Egrad, eps):
        self.E = E
        self.Egrad = Egrad
        self.eps = eps
        self.momenta_scale = 1

    def H(self, z, r):
        return self.E(z) + 0.5*sum(r**2)

    def leapfrog_update(self, zcur, rcur, eps, Hcur, *args, **kwargs):

        rhalfstep = rcur - 0.5*eps*self.Egrad(zcur, *args, **kwargs)
        znew = zcur + eps*rhalfstep / self.momenta_scale**2
        rnew = rhalfstep - 0.5*eps*self.Egrad(znew, *args, **kwargs)

        if np.any(znew <= 0):
            return zcur, rcur, Hcur
        else:

            Hnew = self.H(znew, rnew)
            A = np.exp(Hnew - Hcur)

            if np.random.uniform() <= A:
                return znew, rnew, Hnew

            else:
                return znew, rcur, Hcur

    def momenta_update(self, rcur, *args, **kwargs):
        return np.random.normal(scale=self.momenta_scale,
                                size=rcur.size)

    def sample(self, zcur, n_steps=10, *args, **kwargs):
        zcur = np.array(zcur)
        rcur = np.random.normal(size=zcur.size)
        eps = np.random.choice([-1., 1.])*self.eps
        Hcur = self.H(zcur, rcur)
        for k in range(n_steps):
            zcur, rcur, Hcur = self.leapfrog_update(zcur, rcur, eps, Hcur)
        return zcur
