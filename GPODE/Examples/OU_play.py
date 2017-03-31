import numpy as np
import scipy.linalg
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def dXdt(X, t):
    return -.4*X**2 + np.cos(2*np.pi*t)

fig = plt.figure()
ax = fig.add_subplot(111)

tt = np.linspace(0., .5, 111)

#ind1 = np.arange(0, tt.size, 15)
#ind2 = np.arange(5, tt.size, 7)
#ind3 = np.arange(4, tt.size, 7)

ind1 = [0, 55, 110]
ind2 = [0, 55, 110]
ind3 = [0, 55, 110]


t1 = tt[ind1]
t2 = tt[ind2]
t3 = tt[ind3]

sol1 = odeint(dXdt, 0.3, tt)
sol2 = odeint(dXdt, 0.6, tt)
sol3 = odeint(dXdt, -0.2, tt)

y1 = sol1[ind1]
y2 = sol2[ind2]
y3 = sol3[ind3]

ax.plot(tt, sol1, 'k-.', alpha=0.8)
ax.plot(tt, sol2, 'k-.', alpha=0.8)
ax.plot(tt, sol3, 'k-.', alpha=0.8)

ax.plot(t1, y1, 'bo')
ax.plot(t2, y2, 'bo')
ax.plot(t3, y3, 'bo')
       

def ks(x0,y0):
    return np.exp(-(x0-y0)**2)

def kt(s, t):
    return np.exp(-5*(s-t)**2)

from gp_ivp import interp_par
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

tnew = tt.copy()
for x0 in np.linspace(-.5, 1.1, 15):
    z = interp_par(x0, tnew, [t1, t2, t3], [0.3, 0.6, -0.2], [y1, y2, y3], kt, ks)
    ax.plot(tnew, z, 'r-' ,alpha=0.2)

    sol = odeint(dXdt, x0,  tt)

    ax2.plot(tt, sol, 'k-.', alpha=0.5)
    ax2.plot(tt, z, 'r-', alpha=0.5)

print interp_par(0.4, np.array([0.6]), [t1, t2, t3],
                 [0.3, 0.6, -0.2], [y1, y2, y3], kt, ks)

plt.show()
