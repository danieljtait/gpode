import numpy as np
import matplotlib.pyplot as plt

from gp_src import ksqexp, GaussianProcess


gp1 = GaussianProcess(ksqexp, 5.)
tt = np.linspace(0., 1., 11)

rf = gp1.sim(tt)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tt, rf, 'o')

plt.show()
