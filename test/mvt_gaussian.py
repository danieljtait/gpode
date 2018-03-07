import numpy as np
from scipy.stats import multivariate_normal

# Returns the Expected value of
#
# res[i, j] = E X_i^p X_j^p ]
def Ex_monomial(degree, mean, cov):

    if degree == 1:
        return mean

    elif degree == 2:
        return cov + np.outer(mean, mean)

    elif degree == 4:
        dm = np.diag(mean)
        cii = np.diag(cov)
        
        expr0 = mvtnorm_central_monomial(4, cov)
        expr1 = np.outer(cii, mean**2)
        expr2 = 4*np.dot(dm, np.dot(cov, dm))
        expr3 = np.outer(mean**2, cii)

        return expr0 + expr1 + expr2 + expr3 + np.outer(mean**2, mean**2)


##
#
def mvtnorm_central_monomial(degree, cov):

    if degree == 2:
        return cov

    if degree == 4:
        return np.outer(np.diag(cov), np.diag(cov)) + 2*cov**2

mean = np.array([0.5, 1.1])
cov = np.array([[1.3, 0.4],
                [0.4, 1.1]])

NSIM = 2000000
rz = multivariate_normal.rvs(mean=mean,
                             cov=cov,
                             size=NSIM)

i = 0
j = 0
X = rz[:,i]**2*rz[:, j]**2
print(np.mean(X))

print(Ex_monomial(4, mean, cov))

    
    
