import numpy as np
import matplotlib.pyplot as plt
from gp_src import ksqexp, GaussianProcess


gp1 = GaussianProcess(ksqexp, kernel_par=5.)
tp = np.linspace(0., 3.5, 5)

# Simulate from this Gaussian Process
rf = gp1.sim(tp)


# Fit and interpolate
gp1.interp_fit(tp, rf)


tt = np.linspace(tp[0], 5, 150)
fv  = [gp1.interp_evalf(t,wVar=True) for t in tt]
fv = np.array(fv)

# Plot it
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tp, rf, 'bo')
ax.plot(tt, fv[:,0], 'b-')

sd = np.sqrt(fv[:,1])
ax.fill_between(tt, fv[:,0] + sd, fv[:,0] - sd, facecolor='blue', alpha=0.1)

plt.show()
