import numpy as np


def interp_par(x0, t0,
               Ts, X0s, Xs,
               kt, ks):

    S00, T00 = np.meshgrid(t0, t0)
    S01, T01 = np.meshgrid(t0, Ts[0])

    x0_i = x0*np.ones(t0.size)
    y0_i = X0s[0]*np.ones(Ts[0].size)
    X0_i, Y0_i = np.meshgrid(x0_i, y0_i)

    C01_t = kt(S01.T.ravel(), T01.T.ravel()).reshape(t0.size, Ts[0].size)
    C01_s = ks(X0_i.T.ravel(), Y0_i.T.ravel()).reshape(t0.size, Ts[0].size)
    
    C00 = kt(S00.ravel(), T00.ravel()).reshape(t0.size, t0.size)
    C01 = C01_t * C01_s

    for i in range(len(Ts)-1):
        S01, T01 = np.meshgrid(t0, Ts[i+1])

        x0_i = x0*np.ones(t0.size)
        y0_i = X0s[i+1]*np.ones(Ts[i+1].size)
        X0_i, Y0_i = np.meshgrid(x0_i, y0_i)

        C0i_t = kt(S01.T.ravel(), T01.T.ravel()).reshape(t0.size, Ts[i+1].size)
        C0i_s = ks(X0_i.T.ravel(), Y0_i.T.ravel()).reshape(t0.size,Ts[i+1].size)
        
        C01 = np.column_stack((C01, C0i_t*C0i_s))

    aa = []
    x0s = []
    for i in range(len(Ts)):
        aa = np.concatenate(( aa, Xs[i][:,0]))
        x0s = np.concatenate(( x0s, X0s[i]*np.ones(Ts[i].size) ))

    x0s = np.array(x0s)
    tt = np.concatenate(([t for t in Ts]))
    

    S, T = np.meshgrid(tt, tt)
    X01, X02 = np.meshgrid(x0s, x0s)

    C11 = kt(S.ravel(), T.ravel()).reshape(tt.size, tt.size) * ks(X01.ravel(), X02.ravel()).reshape(x0s.size, x0s.size)
    C11inv = np.linalg.inv(C11)

    return np.dot(C01, np.dot(C11inv, aa))

