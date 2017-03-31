import numpy as np
from gp_ode_src import LatentVelocity
from gp_src import ksqexp, GaussianProcess
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def dXdt(X,t=0):
    return np.array([-X[1],X[0]])

gp1 = GaussianProcess(ksqexp, kernel_par=5.)
gp2 = GaussianProcess(ksqexp, kernel_par=3.)

T = 2*np.pi
tt = np.linspace(0., T, 111)
sol = odeint(dXdt, [1., 0.], tt)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sol[:,0], sol[:,1], 'k-.')

tp = np.linspace(0., T, 5)

for nt in range(5):
    LV = LatentVelocity(dXdt, [gp1, gp2])
    for p in range(2):
        LV.sim_latent(p, tp, set=True)
                    
    rsol = odeint(LV.dXdt, [1., 0.], tt)
    ax.plot(rsol[:,0], rsol[:,1], alpha=0.2)






plt.show()
