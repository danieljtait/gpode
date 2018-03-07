import numpy as np
from VarMLFM_adapGrad import VarMLFM_adapGrad
from scipy.integrate import odeint
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

A0 = np.array([[0.0, 0.],
               [0., 0.]])
A1 = np.array([[0., -1.],
               [1., 0.]])

tt = np.linspace(0., 2., 5)
x0 = np.array([1., 0.])

def g1(t):
    return np.sin(t)

sol = odeint(lambda x, t: np.dot(A0 + g1(t)*A1, x), x0, tt)

x_gp_par = ([1., 5.],
            [1., 5.])
g_gp_par = ([1., 1.],)


vobj = VarMLFM_adapGrad([A0, A1],
                        tt,
                        sol,
                        [0.1, 0.1],
                        [.1, .1],
                        x_gp_par)

fig = plt.figure()
ax = fig.add_subplot(111)

for nt in range(3):
    vobj._update_X_var_dist()    
    vobj._update_G_var_dist()


    ax.plot(tt, vobj._X_means[0], 'k-.', alpha=0.2)
ax.plot(tt, sol[:, 0], 'sr')
plt.show()

