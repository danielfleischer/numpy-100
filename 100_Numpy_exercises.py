import numpy as np

a.size
a.itemsize
a.size * a.itemsize

a = np.arange(10,50)

a = a[::-1]

a = np.arange(0,9).reshape(3,3)

np.nonzero(a)

np.eye(3)

print(a.min(), a.max())

print(a.mean())

a = np.zeros((10,10))
a = np.pad(a, 1, constant_values=1)

a = np.ones((10,10))
a[1:-1,1:-1] = 0

np.diag([1,2,3,4], -1)

a[::2,1::2] = 1
a[1::2,::2] = 1

np.unravel_index(99, (6,7,8))

np.tile([[1,0],[0,1]], (4,4))

z = (a - a.mean())/a.std()

np.dtype([("r", np.ubyte),("g", np.ubyte),("b", np.ubyte),("a", np.ubyte)])

a = np.random.rand(5,3)
b = np.random.rand(3,2)
a @ b

a[(a > 3) & (a < 8)] *= -1

np.where(a > 0, np.ceil(a), np.floor(a))

np.sqrt(-1)
np.emath.sqrt(-1)

date = np.datetime64
now = date('today')
np.timedelta64(1, 'D') + now
np.timedelta64(-1, 'D') + now
np.arange('2016-07', '2016-08', dtype='datetime64[D]')

np.trunc(a)


np.tile(np.arange(0,5),(5,1))

np.fromiter(range(10), int)

np.linspace(0,1,10)

a.sort()


np.add.reduce(a)


np.allclose(a,b)
np.array_equal(a,b)

a[a.argmax()] = 0

Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)


np.subtract.outer(x,y)


arr = np.add.outer(x,y)


Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])

Z[0]['position']['x']

import scipy.spatial
scipy.spatial.distance.cdist(x,x)

X,Y = np.atleast_2d(x[:,0], x[:,1])
np.sqrt((X-X.T)**2 + (Y-Y.T)**2)

a[:] = a.view(int)

from io import StringIO

s = StringIO("""1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11""")

np.genfromtxt(s,delimiter=',', filling_values=0)


list(np.ndenumerate(np.random.rand(3,4)))
list(np.ndindex((3,4)))


a - a.mean(1).reshape(-1,1)
a - a.mean(1,keepdims=True)


a.sum((-1,-2))

np.diag(A @ B)
np.einsum('ij, ji -> i',A,B)  # diagonal

np.einsum('ij, ji',A,B)  # Trace of result
