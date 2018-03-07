import numpy as np
import matplotlib.pyplot as plt

mlist = np.array([0, 3, 5, 5, 10])
Yorig = np.array([0., 1., 2., 3., 4.])
Ynew = Yorig.copy()
for nt, i in enumerate(mlist):
    Ni = sum(i==mlist)
    if Ni > 1:
        i_where = np.where(i == mlist)[0]
        mlist = np.delete(mlist, i_where[1:])

        yavg = np.mean(Ynew[i_where])

        Ynew = np.delete(Ynew, i_where[1:])
        Ynew[nt] = yavg
        
        
print(mlist)
print(Ynew)
    

"""
N = 5
T = 5

ts = np.concatenate(([0.],
                     np.random.uniform(0, T, size=N-2),
                     [T]))
ts = np.sort(ts)
y = np.random.normal(size=ts.size)

def snap_ts(ts, h):
    tmesh = np.arange(ts[0], ts[-1]+h, h)

    inds = []
    for t in ts:
        dt = abs(t-tmesh)
        dt_min = np.min(dt)

        ind = np.where(dt == dt_min)
        inds.append(ind)

    

    return inds, tmesh

def snap_Y(ts, Y, h):
    pass

h = 0.5
inds, tmesh = snap_ts(ts, h)

new_ts = []
for n, ind in enumerate(inds):
    new_ts.append(tmesh[ind[0]])


inds = [ind[0][0] for ind in inds]
print(inds)

tn = []
yn = []
Y = y.copy()
ycopy = y.copy()
for i, y in zip(inds, ycopy):
    if sum(i == inds) > 1:
        print("!")
        # handle repeated point
        rp_point_inds = np.where(i == inds)[0]
        ynew = np.mean(Y[rp_point_inds])

        tn.append(tmesh[i])
        yn.append(yn)
        # points are handled so remove them
        for ind in rp_point_inds:
            ycopy = np.delete(ycopy, ind)

        inds.remove(i)            
    else:
        tn.append(tmesh[i])
        yn.append(y)

print(tn)
print(yn)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(ts, Y, 'ko')
ax.plot(new_ts, Y, 'r+-', alpha=0.2)

#ax.plot(tn, yn, 'g-')

plt.show()
"""
