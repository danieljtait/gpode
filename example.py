import numpy as np
import gpode
from scipy.integrate import odeint
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

A0 = np.array([[0.0, 0.],
               [0., 0.]])
A1 = np.array([[0., -1.],
               [1., 0.]])

N = 10

tt = np.linspace(0., 2., N)
x0 = np.array([1., 0.])

def g1(t):
    return np.sin(t)

sol = odeint(lambda x, t: np.dot(A0 + g1(t)*A1, x), x0, tt)

x_gp_par = ([1., 2.],
            [1., 2.])
g_gp_par = ([1., 1.],)


vobj = gpode.varmlfm_adapgrad([A0, A1],
                              tt,
                              sol,
                              [0.025, 0.025],
                              [.1, .1],
                              x_gp_par,
                              g_gp_par)

fig = plt.figure()
ax = fig.add_subplot(111)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

nsim = 10
for nt in range(nsim):
    vobj._update_g_var_dist()    
    vobj._update_x_var_dist()

    a = (nt+1)/nsim

    ax.plot(tt, vobj._X_means[0], 'k-', alpha=a)
#    ax2.plot(tt, vobj._G_means[0], 'k-', alpha=a)

ax.plot(tt, sol[:, 0], 'sr')
ax2.plot(tt, g1(tt), 'sr')

gvar = vobj._G_covars[(0, 0)]
sd = np.sqrt(np.diag(gvar))

ax2.errorbar(tt, vobj._G_means[0], yerr=sd, fmt='o')

plt.show()

