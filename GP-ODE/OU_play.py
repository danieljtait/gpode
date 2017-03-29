import numpy as np


A = np.random.normal(size=4).reshape(2,2)

print A

l, U = np.linalg.eig(A)
D = np.diag(l)

print D
print U

Uinv = np.linalg.inv(U)


print np.dot(Uinv, np.dot(A, U))
