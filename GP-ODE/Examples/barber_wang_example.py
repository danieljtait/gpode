import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from barber_wang import gp_ode_bw, ksqexp


# Model function
def dXdt(X, t, par):
    return np.array([X[1], -par*par*X[0]])

# True parameter
p0 = 0.79
s0 = 0.5

# Simulate some data
N = 11
tt = np.linspace(0., 2*np.pi, N)
np.random.seed(11)
X0 = odeint(dXdt, [1., 0.], tt, args=(p0,))
Y = X0 + np.random.normal(scale=s0,size=X0.size).reshape(X0.shape)


# Set up the GP model
def dXdt_(X, t):
    return dXdt(X, t, p0)

class model_par_prior:
    def __init__(self, pdf, rvs, hyperparams):
        self.pdf_ = pdf




kernels = (ksqexp, ksqexp)
kpar = ([1., 5.], [0.5, 2.4])

gp0 = gp_ode_bw(F=dXdt_, kernels=kernels, kernels_par = kpar)
gp1 = gp_ode_bw(F=dXdt_, kernels=kernels, kernels_par = kpar)

Xm = np.column_stack((np.mean(Y[:,0])*np.ones(N), np.mean(Y[:,1])*np.ones(N)))

gp0.set_latent_states(tt, X0)
gp1.set_latent_states(tt, Xm)

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(tt, X0[:,0], 'k+')
ax.plot(tt, X0[:,1], 'k+')

#ax.plot(X0[:,0], X0[:,1], 'k+')
#ax.plot(Y[:,0], Y[:,1], 'o')

tt_l = np.linspace(0., 2*np.pi, 10*N)

z0 = [gp0.interp_latent_states_evalf(s, 0) for s in tt_l]
z1 = [gp0.interp_latent_states_evalf(s, 1) for s in tt_l]

x0 = [gp1.interp_latent_states_evalf(s, 0) for s in tt_l]
x1 = [gp1.interp_latent_states_evalf(s, 1) for s in tt_l]

ax.plot(tt_l, z0, 'r-')
ax.plot(tt_l, z1, 'r-')

ax.plot(tt_l, x0, 'b-')
ax.plot(tt_l, x1, 'b-')


fig2 = plt.figure()
ax1 = fig2.add_subplot(121)
ax2 = fig2.add_subplot(122)

ax1.plot(tt, Y[:,0], 'ko')
ax2.plot(tt, Y[:,1], 'ko')

plt.show()
