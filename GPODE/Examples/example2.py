import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gp_src import ksqexp, GaussianProcess


# Initial Condition
x0 = 1.
ss = np.linspace(0., 5, 51)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for i in range(5):

    gp1 = GaussianProcess(ksqexp, kernel_par=5.)
    tp = np.linspace(0., 3.5, 15)
    rf = gp1.sim(tp)

    # Interpolate the discrete realisation of the GP
    gp1.interp_fit(tp, rf)

    def dXdt(X, t=0):
        return -.5*X + gp1.interp_evalf(t)

    sol = odeint(dXdt, x0, ss)

    ax1.plot(tp, rf)
    ax2.plot(ss, sol)

plt.show()
    
