###
#
# Uses the variational fit of the gpode to
#
import numpy as np
from gpode import varmlfm_adapgrad as varmlfm
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# true latent force function, cosine modulated
# gaussian impulse
def g(t):
    dt = t - 2.5  # center the impulse peak
    return np.cos(2*dt)*np.exp(-dt**2)

# Make some data
Ndata = 11
Ndense = 50
t0, T = (0, 5.)

tt_dense = np.linspace(t0, T, Ndense)

# initial condition
x0 = np.array([1., 0.])

# model matrix, corresponds to complex multiplication z' = iz
A = np.array([[-1., 0.],
              [1., 0.]])

sol = odeint(lambda x, t: np.dot(A*g(t), x), x0, tt_dense)

# Subsample the numerical solution to get some data
sample_inds = np.linspace(0, tt_dense.size-1, Ndata, dtype=np.intp)
tt = tt_dense[sample_inds]
Y = sol[sample_inds]

# Initalise the model
x_gp_par = ([1., .25],
            [1., .25])

g_gp_par = ([1., 1.],)

vobj = varmlfm([np.zeros((2, 2)), A],
               tt,
               Y,
               [0.1, 0.1],
               [0.1, 0.1],
               x_gp_par,
               g_gp_par)

n_iter = 10
for nt in range(n_iter):
    vobj._update_g_var_dist()
    vobj._update_x_var_dist()



# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111)
for i, EX in enumerate(vobj._X_means):
    sd = np.sqrt(np.diag(vobj._X_covars[(i, i)]))
    ax.errorbar(tt, EX, yerr=2*sd,
                fmt='o', capsize=5)

    
ax.plot(tt_dense, sol, 'k-', alpha=0.2)

fig2 = plt.figure()
ax = fig2.add_subplot(111)

EG = vobj._G_means[0]
CovG = vobj._G_covars[(0, 0)]
sd = np.sqrt(np.diag(CovG))

ax.errorbar(tt, EG, yerr=2*sd, fmt='o',
            capsize=5)

ax.plot(tt_dense, g(tt_dense), 'k-', alpha=0.2)


plt.show()

    
