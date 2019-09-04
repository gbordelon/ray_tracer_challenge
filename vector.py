import numpy as np
from functools import lru_cache

CACHE_SIZE=2048

class ImmutablePointOrVector(object):
    def __init__(self, ndarr):
        self.arr = ndarr
        self.shape = ndarr.shape

    def compare(self, other):
        return all(np.isclose(self.arr, other.arr))

    def __getitem__(self, x):
        return self.arr[x]

    def __add__(self, b):
        return ImmutablePointOrVector(self.arr + b.arr)

    def __sub__(self, b):
        return ImmutablePointOrVector(self.arr - b.arr)

    def __mul__(self, b):
        return ImmutablePointOrVector(self.arr * b)

    def __truediv__(self, b):
        return ImmutablePointOrVector(self.arr / b)

    def __neg__(self):
        return ImmutablePointOrVector(-self.arr)

    def __repr__(self):
        return self.arr.__repr__()

    #def __setitem__(self, key, value):
    #    self.__setattr__(key, value)

    #def __delitem__(self, key):
    #    self.__delattr__(key)

@lru_cache(maxsize=CACHE_SIZE)
def point_or_vector(x, y, z, w):
    """
    Return a tuple represented by a numpy array
    w should be 0.0 (for a vector) or 1.0 (for a point)

    >>> point_or_vector(1.0, 2.0, 3.0, 1.0)
    array([1., 2., 3., 1.])

    >>> point_or_vector(1.0, 2.0, 3.0, 0.0)
    array([1., 2., 3., 0.])

    Adding two tuples:
    >>> a1 = point_or_vector(3, -2, 5, 1)
    >>> a2 = point_or_vector(-2, 3, 1, 0)
    >>> a1 + a2
    array([1., 1., 6., 1.])

    Negating a tuple:
    >>> a = point_or_vector(1, -2, 3, -4)
    >>> -a
    array([-1.,  2., -3.,  4.])

    Multiply a tuple by a scalar:
    >>> a = point_or_vector(1, -2, 3, -4)
    >>> a * 3.5
    array([  3.5,  -7. ,  10.5, -14. ])

    Multiply a tuple by a fraction:
    >>> a = point_or_vector(1, -2, 3, -4)
    >>> a * 0.5
    array([ 0.5, -1. ,  1.5, -2. ])

    Dividing a tuple by a scalar:
    >>> a = point_or_vector(1, -2, 3, -4)
    >>> a / 2
    array([ 0.5, -1. ,  1.5, -2. ])
    """
    return ImmutablePointOrVector(np.array([x,y,z,w], dtype=np.float64))

def point(x, y, z):
    """
    >>> point(4, -4, 3)
    array([ 4., -4.,  3.,  1.])

    >>> point_or_vector(1.0, 2.0, 3.0, 1.0) == point(1.0, 2.0, 3.0)
    True

    >>> point_or_vector(1.0, 2.0, 3.0, 0.0) == point(1.0, 2.0, 3.0)
    False

    """
    return point_or_vector(x,y,z,1.0)

def vector(x, y, z):
    """
    >>> vector(4, -4, 3)
    array([ 4., -4.,  3.,  0.])

    >>> point_or_vector(1.0, 2.0, 3.0, 1.0)
    array([1., 2., 3., 1.])

    >>> point_or_vector(1.0, 2.0, 3.0, 0.0)
    array([1., 2., 3., 0.])

    Subtracting two vectors:
    >>> v1 = vector(3,2,1)
    >>> v2 = vector(5,6,7)
    >>> v1 - v2
    array([-2., -4., -6.,  0.])

    Subtracting a vector from the zero vector:
    >>> zero = vector(0,0,0)
    >>> v = vector(1,-2,3)
    >>> zero - v
    array([-1.,  2., -3.,  0.])
    """
    return point_or_vector(x,y,z,0.0)

def magnitude(v):
    """
    >>> v = vector(1,0,0)
    >>> magnitude(v) == 1
    True
    >>> v = vector(0,1,0)
    >>> magnitude(v) == 1
    True
    >>> v = vector(0,0,1)
    >>> magnitude(v) == 1
    True
    >>> v = vector(1,2,3)
    >>> magnitude(v) == np.sqrt(14)
    True
    >>> v = vector(-1,-2,-3)
    >>> magnitude(v) == np.sqrt(14)
    True
    >>> p = point(1,2,3)
    >>> magnitude(p)
    Traceback (most recent call last):
    ...
    ValueError: Only use this function with vectors.
    >>> t = point_or_vector(1,2,3,-2)
    >>> magnitude(p)
    Traceback (most recent call last):
    ...
    ValueError: Only use this function with vectors.
    >>> t = np.array([[1,2,3,0,1,2,3,4]])
    >>> magnitude(p)
    Traceback (most recent call last):
    ...
    ValueError: Only use this function with vectors.
    """
    if len(v.arr) != 4 or v[3] != 0.0:
        raise ValueError("Only use this function with vectors.")
    return np.sqrt(np.sum(np.square(v.arr)))

@lru_cache(maxsize=CACHE_SIZE)
def normalize(v):
    """
    >>> v = vector(4,0,0)
    >>> normalize(v)
    array([1., 0., 0., 0.])

    >>> v = vector(1,2,3)
    >>> sq = np.sqrt(1+4+9)
    >>> normalize(v)
    array([0.26726124, 0.53452248, 0.80178373, 0.        ])
    """

    return v * (1.0 / magnitude(v))

@lru_cache(maxsize=CACHE_SIZE)
def dot(a, b):
    """
    >>> v1 = vector(3, 6, 9)
    >>> v2 = vector(2, 3, 4)
    >>> dot(v1, v2) == 3 * 2 + 6 * 3 + 9 * 4
    True
    """
    return np.vdot(a.arr,b.arr)

# try np.cross(a,b) but have to change vectors to 1x3 instead of 1x4
@lru_cache(maxsize=CACHE_SIZE)
def cross(a, b):
    """
    >>> v1 = vector(1,2,3)
    >>> v2 = vector(2,3,4)
    >>> cross(v1, v2) == vector(-1,2,-1)
    True

    >>> v1 = vector(1,2,3)
    >>> v2 = vector(2,3,4)
    >>> cross(v2, v1) == vector(1,-2,1)
    True

    """
    #return np.cross(a,b)

    return vector(a[1] * b[2] - a[2] * b[1],
                  a[2] * b[0] - a[0] * b[2],
                  a[0] * b[1] - a[1] * b[0])

def dummy_1():
    """
    Subtracting two points:
    >>> p1 = point(3,2,1)
    >>> p2 = point(5,6,7)
    >>> p1 - p2
    array([-2., -4., -6.,  0.])

    Subtracting a vector from a point:
    >>> p = point(3,2,1)
    >>> v = vector(5,6,7)
    >>> p - v
    array([-2., -4., -6.,  1.])

    Magnitude of a normalized vector:
    >>> v = vector(1,2,3)
    >>> norm = normalize(v)
    >>> magnitude(norm) == 1
    True
    """
    pass