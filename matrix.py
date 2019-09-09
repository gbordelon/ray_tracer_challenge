from vector import *

from functools import lru_cache
import numpy as np

CACHE_SIZE=65536

class ImmutableMatrix(object):
    def __init__(self, ndarr):
        self.arr = ndarr
        self.shape = ndarr.shape

    def compare(self, other):
        return all([all(row) for row in np.isclose(self.arr, other.arr)])

    def __getitem__(self, x, y):
        return self.arr[x, y]

    def __getitem__(self, x):
        return self.arr[x]

    def __add__(self, b):
        return ImmutableMatrix(self.arr + b.arr)

    def __sub__(self, b):
        return ImmutableMatrix(self.arr - b.arr)

    def __mul__(self, b):
        return matrix_multiply(self, b)

    def __truediv__(self, b):
        # assume divide by float
        return ImmutableMatrix(self.arr / b)

    def __neg__(self):
        return ImmutableMatrix(-self.arr)

    def __repr__(self):
        return self.arr.__repr__()


def matrix(*args):
    if len(args) == 4:
        return matrix2x2(*args)
    elif len(args) == 9:
        return matrix3x3(*args)
    elif len(args) == 16:
        return matrix4x4(*args)
    else:
        return None

@lru_cache(maxsize=CACHE_SIZE)
def matrix4x4(a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4):
    """
    >>> m = matrix(1,2,3,4, 5.5,6.5,7.5,8.5, 9,10,11,12, 13.5,14.5,15.5,16.5)
    >>> m[0][0] == 1.0 and m[1][0] == 5.5 and m[1][2] == 7.5 and m[2][2] == 11 and m[3][0] == 13.5 and m[3][2] == 15.5
    True
    """
    m = np.zeros((4,4), dtype=np.float64)
    m[0,0] = a1
    m[0,1] = a2
    m[0,2] = a3
    m[0,3] = a4
    m[1,0] = b1
    m[1,1] = b2
    m[1,2] = b3
    m[1,3] = b4
    m[2,0] = c1
    m[2,1] = c2
    m[2,2] = c3
    m[2,3] = c4
    m[3,0] = d1
    m[3,1] = d2
    m[3,2] = d3
    m[3,3] = d4
    return ImmutableMatrix(m)

def matrix4helper(r1, r2, r3, r4):
    return ImmutableMatrix(np.array([r1, r2, r3, r4]))

@lru_cache(maxsize=CACHE_SIZE)
def matrix4x4identity():
    return matrix4x4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1)

class Transform(object):
    @classmethod
    def translate(cls, x, y, z):
        return matrix(1,0,0,x, 0,1,0,y, 0,0,1,z, 0,0,0,1)

    @classmethod
    def scale(cls, x, y, z):
        return matrix(x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1)

    @classmethod
    def rotate_x(cls, rad):
        return matrix(1,0,0,0, 0,np.cos(rad),-np.sin(rad),0, 0,np.sin(rad),np.cos(rad),0, 0,0,0,1)

    @classmethod
    def rotate_y(cls, rad):
        return matrix(np.cos(rad),0,np.sin(rad),0, 0,1,0,0, -np.sin(rad),0,np.cos(rad),0, 0,0,0,1)

    @classmethod
    def rotate_z(cls, rad):
        return matrix(np.cos(rad),-np.sin(rad),0,0, np.sin(rad),np.cos(rad),0,0, 0,0,1,0, 0,0,0,1)

    @classmethod
    def shear(cls, xy, xz, yx, yz, zx, zy):
        return matrix(1,xy,xz,0, yx,1,yz,0, zx,zy,1,0, 0,0,0,1)

    @classmethod
    def from_yaml(cls, obj): # assume a list of lists
        xlations = []
        for l in obj:
            if l[0] == 'translate':
                xlations.append(cls.translate(l[1], l[2], l[3]))
            elif l[0] == 'scale':
                xlations.append(cls.scale(l[1], l[2], l[3]))
            elif l[0] == 'rotate-x':
                xlations.append(cls.rotate_x(l[1]))
            elif l[0] == 'rotate-y':
                xlations.append(cls.rotate_y(l[1]))
            elif l[0] == 'rotate-z':
                xlations.append(cls.rotate_z(l[1]))
            elif l[0] == 'shear':
                xlations.append(cls.shear(l[1], l[2], l[3], l[4], l[5], l[6]))
            else:
                raise ValueError('Unknown translation type: {}'.format(l[0]))
        acc = matrix4x4identity()
        for xlation in xlations:
            tmp = xlation * acc
            acc = tmp
        return acc

def matrix3x3(a1, a2, a3, b1, b2, b3, c1, c2, c3):
    """
    >>> m = matrix(-3,5,0, 1,-2,-7, 0,1,1)
    >>> m[0][0] == -3.0 and m[1][1] == -2.0 and m[2][2] == 1
    True
    """
    return matrix3helper(np.array([a1,a2,a3], dtype=np.float64),
                         np.array([b1,b2,b3], dtype=np.float64),
                         np.array([c1,c2,c3], dtype=np.float64))

def matrix3helper(r1, r2, r3):
    return ImmutableMatrix(np.array([r1, r2, r3]))

def matrix2x2(a1, a2, b1, b2):
    """
    >>> m = matrix(-3, 5, 1, -2)
    >>> m[0][0] == -3.0 and m[0][1] == 5 and m[1][0] == 1 and m[1][1] == -2
    True
    """
    return matrix2helper(np.array([a1,a2], dtype=np.float64),
                         np.array([b1,b2], dtype=np.float64))

def matrix2helper(r1, r2):
    return ImmutableMatrix(np.array([r1, r2]))

@lru_cache(maxsize=CACHE_SIZE)
def matrix_multiply(a, b):
    """
    Matrix multiplication:
    >>> m1 = matrix(1,2,3,4, 5,6,7,8, 9,8,7,6, 5,4,3,2)
    >>> m2 = matrix(-2, 1, 2, 3, 3, 2, 1, -1, 4, 3, 6, 5, 1, 2, 7, 8)
    >>> m3 = matrix(20, 22, 50, 48, 44, 54, 114, 108, 40, 58, 110, 102, 16, 26, 46, 42)
    >>> matrix_multiply(m1, m2)
    array([[ 20.,  22.,  50.,  48.],
           [ 44.,  54., 114., 108.],
           [ 40.,  58., 110., 102.],
           [ 16.,  26.,  46.,  42.]])
    """
    arr = np.matmul(a.arr,b.arr)
    if len(b.arr.shape) == 1:
        return ImmutablePointOrVector(arr)
    return ImmutableMatrix(arr)

    """
    if len(b.shape) == 1:
        m = point_or_vector(0,0,0,0)
    else:
        m = matrix(0,0,0,0,
                      0,0,0,0,
                      0,0,0,0,
                      0,0,0,0)

    for i in range(a.shape[0]):
        if len(b.shape) == 1:
            m[i] = a[i][0] * b[0] +\
                   a[i][1] * b[1] +\
                   a[i][2] * b[2] +\
                   a[i][3] * b[3]
        else:
            for j in range(b.shape[0]):
                m[i][j] = a[i][0] * b[0][j] +\
                          a[i][1] * b[1][j] +\
                          a[i][2] * b[2][j] +\
                          a[i][3] * b[3][j]
    return m
    """

@lru_cache(maxsize=CACHE_SIZE)
def transpose(a):
    """
    >>> m1 = matrix(0,9,3,0, 9,8,0,8, 1,8,5,3, 0,0,5,8)
    >>> m2 = matrix(0,9,1,0, 9,8,8,0, 3,0,5,5, 0,8,3,8)
    >>> transpose(m1)
    array([[0., 9., 1., 0.],
           [9., 8., 8., 0.],
           [3., 0., 5., 5.],
           [0., 8., 3., 8.]])

    """
    b = np.transpose(a.arr)
    return ImmutableMatrix(b)

def dummy_3():
    """
    Matrix equality:
    >>> m1 = matrix(1,2,3,4, 5.5,6.5,7.5,8.5, 9,10,11,12, 13.5,14.5,15.5,16.5)
    >>> m2 = matrix(1,2,3,4, 5.5,6.5,7.5,8.5, 9,10,11,12, 13.5,14.5,15.5,16.5)
    >>> m1.compare(m2)
    True

    Matrix inequality:
    >>> m1 = matrix(1,2,3,4, 5.5,6.5,7.5,8.5, 9,10,11,12, 13.5,14.5,15.5,16.5)
    >>> m2 = matrix(1,2,3,99, 5.5,6.5,88,8.5, 9,77,11,12, -66,14.5,15.5,16.5)
    >>> m1.compare(m2)
    False

    Matrix identity multiplication:
    >>> ident = matrix4x4identity()
    >>> m = matrix(0,1,2,4, 1,2,4,8, 2,4,8,16, 4,8,16,32)
    >>> m * ident
    array([[ 0.,  1.,  2.,  4.],
           [ 1.,  2.,  4.,  8.],
           [ 2.,  4.,  8., 16.],
           [ 4.,  8., 16., 32.]])

    Multiply identity matrix by tuple:
    >>> ident = matrix4x4identity()
    >>> a = ImmutablePointOrVector(np.array([1,2,3,4]))
    >>> ident * a
    array([1., 2., 3., 4.])

    Transpose identiy matrix:
    >>> ident = matrix4x4identity()
    >>> transpose(ident)
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])
    """
    pass

@lru_cache(maxsize=CACHE_SIZE)
def determinant(a):
    """
    >>> m = matrix(1,5, -3,2)
    >>> determinant(m) == 17
    True

    >>> m = matrix(1,2,6, -5,8,-4, 2,6,4)
    >>> np.isclose(cofactor(m, 0, 0),56) and np.isclose(cofactor(m, 0, 1),12) and np.isclose(cofactor(m, 0, 2),-46) and np.isclose(determinant(m),-196)
    True

    >>> m = matrix(-2,-8,3,5, -3,1,7,3, 1,2,-9,6, -6,7,7,-9)
    >>> np.isclose(cofactor(m, 0, 0),690) and np.isclose(cofactor(m, 0, 1),447) and np.isclose(cofactor(m, 0, 2),210) and np.isclose(cofactor(m, 0, 3),51) and np.isclose(determinant(m),-4071)
    True
    """

    """
    if a.shape[0] == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]

    det = 0
    for col in range(a.shape[0]):
        det += a[0][col] * cofactor(a, 0, col)
    return det
    """
    return np.linalg.det(a.arr)

def submatrix(a, row_skip, column_skip):
    """
    >>> m = matrix(1,5,0, -3,2,7, 0,6,-3)
    >>> s = submatrix(m, 0, 2)
    >>> s
    array([[-3.,  2.],
           [ 0.,  6.]])

    >>> m = matrix(-6,1,1,6, -8,5,8,6, -1,0,8,2, -7,1,-1,1)
    >>> s = submatrix(m, 2, 1)
    >>> s
    array([[-6.,  1.,  6.],
           [-8.,  8.,  6.],
           [-7., -1.,  1.]])
    """
    dim = a.shape[0]
    if dim == 3:
        m = matrix(0,0, 0,0)
    elif dim == 4:
        m = matrix(0,0,0, 0,0,0, 0,0,0)

    row = 0
    for i in range(dim):
        if i == row_skip:
            continue
        column = 0
        for j in range(dim):
            if j == column_skip:
                continue
            m[row][column] = a[i][j]
            column += 1
        row += 1
    return m

def minor(a, row_skip, column_skip):
    """
    >>> m1 = matrix(3,5,0, 2,-1,-7, 6,-1,5)
    >>> m2 = submatrix(m1, 1, 0)
    >>> np.isclose(determinant(m2),25) and np.isclose(minor(m1, 1, 0),25)
    True
    """
    return determinant(submatrix(a, row_skip, column_skip))

def cofactor(a, row_skip, column_skip):
    """
    >>> m1 = matrix(3,5,0, 2,-1,-7, 6,-1,5)
    >>> np.isclose(minor(m1, 0, 0),-12) and np.isclose(cofactor(m1, 0, 0),-12) and np.isclose(minor(m1, 1, 0),25) and np.isclose(cofactor(m1, 1, 0),-25)
    True
    """
    m = minor(a, row_skip, column_skip)
    if row_skip + column_skip % 2 == 0:
        return m
    return np.negative(m)

@lru_cache(maxsize=CACHE_SIZE)
def is_invertible(a):
    """
    >>> m = matrix(6,4,4,4, 5,5,7,6, 4,-9,3,-7, 9,1,7,-6)
    >>> is_invertible(m)
    True

    >>> m = matrix(-4,2,-2,-3, 9,6,2,6, 0,-5,1,-5, 0,0,0,0)
    >>> is_invertible(m)
    False
    """
    return determinant(a) != 0.0

@lru_cache(maxsize=CACHE_SIZE)
def inverse(a):
    """
    >>> a = matrix(-5,2,6,-8, 1,-5,1,8, 7,7,-6,-7, 1,-3,7,4)
    >>> b = inverse(a)
    >>> np.isclose(determinant(a),532) and np.isclose(cofactor(a, 2, 3),-160) and np.isclose(b[3][2],-160.0/532.0) and np.isclose(cofactor(a, 3, 2),105) and np.isclose(b[2][3],105.0/532.0)
    True
    >>> b
    array([[ 0.21804511,  0.45112782,  0.2406015 , -0.04511278],
           [-0.80827068, -1.45676692, -0.44360902,  0.52067669],
           [-0.07894737, -0.22368421, -0.05263158,  0.19736842],
           [-0.52255639, -0.81390977, -0.30075188,  0.30639098]])

    >>> a = matrix(8,-5,9,2, 7,5,6,1, -6,0,9,6, -3,0,-9,-4)
    >>> b = inverse(a)
    >>> b
    array([[-0.15384615, -0.15384615, -0.28205128, -0.53846154],
           [-0.07692308,  0.12307692,  0.02564103,  0.03076923],
           [ 0.35897436,  0.35897436,  0.43589744,  0.92307692],
           [-0.69230769, -0.69230769, -0.76923077, -1.92307692]])

    >>> a = matrix(9,3,0,9, -5,-2,-6,-3, -4,9,6,4, -7,6,6,2)
    >>> b = inverse(a)
    >>> b
    array([[-0.04074074, -0.07777778,  0.14444444, -0.22222222],
           [-0.07777778,  0.03333333,  0.36666667, -0.33333333],
           [-0.02901235, -0.1462963 , -0.10925926,  0.12962963],
           [ 0.17777778,  0.06666667, -0.26666667,  0.33333333]])

    Multiplying with an inverse:
    >>> a = matrix(3,-9,7,3, 3,-8,2,-9, -4,4,4,1, -6,5,-1,1)
    >>> b = matrix(8,2,2,2, 3,-1,7,0, 7,0,5,4, 6,-2,0,5)
    >>> b_inv = inverse(b)
    >>> c = matrix_multiply(a,b)
    >>> d = matrix_multiply(c,b_inv)
    >>> isclose(a,d)
    array([[ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True]])
    """
    if not is_invertible(a):
        raise ValueError("Matrix supplied is not invertible.")

    return ImmutableMatrix(np.linalg.inv(a.arr))

def translation(x,y,z):
    """
    >>> tr = translation(5,-3,2)
    >>> p = point(-3,4,5)
    >>> matrix_multiply(tr, p)
    array([2., 1., 7., 1.])

    >>> tr = translation(5,-3,2)
    >>> inv = inverse(tr)
    >>> p = point(-3,4,5)
    >>> matrix_multiply(inv, p)
    array([-8.,  7.,  3.,  1.])

    >>> tr = translation(5,-3,2)
    >>> v = vector(-3,4,5)
    >>> matrix_multiply(tr, v)
    array([-3.,  4.,  5.,  0.])

    """
    tr = matrix4x4identity()
    tr = ImmutableMatrix(np.copy(tr.arr))
    tr[0][3] = x
    tr[1][3] = y
    tr[2][3] = z
    return tr

def scaling(x,y,z):
    """
    >>> tr = scaling(2,3,4)
    >>> p = point(-4,6,8)
    >>> matrix_multiply(tr, p)
    array([-8., 18., 32.,  1.])

    >>> tr = scaling(2,3,4)
    >>> v = vector(-4,6,8)
    >>> matrix_multiply(tr, v)
    array([-8., 18., 32.,  0.])

    >>> tr = scaling(2,3,4)
    >>> inv = inverse(tr)
    >>> v = vector(-4,6,8)
    >>> matrix_multiply(inv, v)
    array([-2.,  2.,  2.,  0.])

    >>> tr = scaling(-1,1,1)
    >>> p = point(2,3,4)
    >>> matrix_multiply(tr, p)
    array([-2.,  3.,  4.,  1.])

    """
    return matrix(x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1)

def rotation_x(rad):
    """
    >>> p = point(0,1,0)
    >>> half_q = rotation_x(np.pi / 4)
    >>> full_q = rotation_x(np.pi / 2)
    >>> isclose(matrix_multiply(half_q, p), point(0, np.sqrt(2)/2, np.sqrt(2)/2))
    array([ True,  True,  True,  True])

    >>> isclose(matrix_multiply(full_q, p), point(0, 0, 1))
    array([ True,  True,  True,  True])

    >>> p = point(0,1,0)
    >>> half_q = rotation_x(np.pi / 4)
    >>> inv = inverse(half_q)
    >>> isclose(matrix_multiply(inv, p), point(0, np.sqrt(2)/2, -np.sqrt(2)/2))
    array([ True,  True,  True,  True])
    """
    return matrix(1,0,0,0, 0,np.cos(rad),-np.sin(rad),0, 0,np.sin(rad),np.cos(rad),0, 0,0,0,1)

def rotation_y(rad):
    """
    >>> p = point(0,0,1)
    >>> half_q = rotation_y(np.pi / 4)
    >>> full_q = rotation_y(np.pi / 2)
    >>> isclose(matrix_multiply(half_q, p), point(np.sqrt(2)/2, 0, np.sqrt(2)/2))
    array([ True,  True,  True,  True])

    >>> isclose(matrix_multiply(full_q, p), point(1, 0, 0))
    array([ True,  True,  True,  True])
    """
    return matrix(np.cos(rad),0,np.sin(rad),0, 0,1,0,0, -np.sin(rad),0,np.cos(rad),0, 0,0,0,1)

def rotation_z(rad):
    """
    >>> p = point(0,1,0)
    >>> half_q = rotation_z(np.pi / 4)
    >>> full_q = rotation_z(np.pi / 2)
    >>> isclose(matrix_multiply(half_q, p), point(-np.sqrt(2)/2, np.sqrt(2)/2, 0))
    array([ True,  True,  True,  True])

    >>> isclose(matrix_multiply(full_q, p), point(-1, 0, 0))
    array([ True,  True,  True,  True])
    """
    return matrix(np.cos(rad),-np.sin(rad),0,0, np.sin(rad),np.cos(rad),0,0, 0,0,1,0, 0,0,0,1)

def shearing(xy, xz, yx, yz, zx, zy):
    """
    >>> tr = shearing(1,0,0,0,0,0)
    >>> p = point(2,3,4)
    >>> matrix_multiply(tr, p)
    array([5., 3., 4., 1.])

    >>> tr = shearing(0,1,0,0,0,0)
    >>> p = point(2,3,4)
    >>> matrix_multiply(tr, p)
    array([6., 3., 4., 1.])

    >>> tr = shearing(0,0,1,0,0,0)
    >>> p = point(2,3,4)
    >>> matrix_multiply(tr, p)
    array([2., 5., 4., 1.])

    >>> tr = shearing(0,0,0,1,0,0)
    >>> p = point(2,3,4)
    >>> matrix_multiply(tr, p)
    array([2., 7., 4., 1.])

    >>> tr = shearing(0,0,0,0,1,0)
    >>> p = point(2,3,4)
    >>> matrix_multiply(tr, p)
    array([2., 3., 6., 1.])

    >>> tr = shearing(0,0,0,0,0,1)
    >>> p = point(2,3,4)
    >>> matrix_multiply(tr, p)
    array([2., 3., 7., 1.])

    """
    return matrix(1,xy,xz,0, yx,1,yz,0, zx,zy,1,0, 0,0,0,1)

def dummy_4():
    """
    >>> p1 = point(1,0,1)
    >>> A = rotation_x(np.pi/2)
    >>> B = scaling(5,5,5)
    >>> C = translation(10,5,7)
    >>> p2 = matrix_multiply(A,p1)

    >>> isclose(p2,point(1,-1,0))
    array([ True,  True,  True,  True])

    >>> p3 = matrix_multiply(B,p2)
    >>> isclose(p3,point(5,-5,0))
    array([ True,  True,  True,  True])

    >>> p4 = matrix_multiply(C,p3)
    >>> p4
    array([15.,  0.,  7.,  1.])

    >>> T = matrix_multiply(matrix_multiply(C,B),A)
    >>> matrix_multiply(T,p1)
    array([15.,  0.,  7.,  1.])
    """
    pass

def isclose(a, b):
    return np.isclose(a.arr, b.arr)