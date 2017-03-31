import numpy as np
from scipy.stats import multivariate_normal
from alv_lfgp import GP_LVFM
from alv_lfgp import make_output_cov, output_mean

D = np.array([1., 0.7])
B = np.zeros(2)
S = np.array([[.1,  0.0],
              [0.0,  .1]])

lv = GP_LVFM(D, B, S, [0.1, 0.1])


tt = np.linspace(0.3, 0.5, 2)
ss = np.linspace(0.24, 0.51, 2)

t1 = tt.copy()
t2 = ss.copy()

Ts = [t1, t2 ]





y0 = np.array([0.3, 0.6])

tt = np.linspace(0.1, 2., 11)

M = np.array([output_mean(t, y0, lv) for t in tt])
m = np.concatenate((M[:,0],M[:,1]))

#m = np.zeros(m.size)


Ts = [tt, tt]


Cov = make_output_cov(Ts, lv)

print np.linalg.eig(Cov)[0]


try:
    L = np.linalg.cholesky(Cov)
    z = np.random.normal(size=m.size)
    z = m + np.dot(L,z)
except:
    print "Cov was singular"
    delta = 0.001
    Cov += np.diag(delta*np.ones(Cov.shape[0]))
    L = np.linalg.cholesky(Cov)
    z = np.random.normal(size=m.size)
    z = m + np.dot(L,z)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tt, z[:tt.size],'bo')
ax.plot(tt, M[:,0],'b-.')

ax.plot(tt, z[tt.size:], 'ro')
ax.plot(tt, M[:,1],'r-.' )

fig2 = plt.figure()
ax = fig2.add_subplot(111)

s = 1.5


tt = np.linspace(0., 5., 211)
ax.plot(tt, [lv.ky(s, t, 0, 0) for t in tt])
ax.plot(tt, [lv.ky(s, t, 1, 1) for t in tt])

plt.show()


