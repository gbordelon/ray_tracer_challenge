from canvas import *
from matrix import *
#from shapes import *
from vector import *

import shapes

import itertools
from multiprocessing import Pool
import numpy as np

EPSILON = 0.000001
BLOCK_SIZE=2

class World(object):
    def __init__(self, lights, contains):
        self.lights = lights
        self.contains = contains

def world():
    """
    >>> w = world()
    >>> w.lights is None
    True
    >>> len(w.contains) == 0
    True
    """
    return World(None, [])

def default_world():
    """
    >>> light = point_light(point(-10, 10, -10), color(1,1,1))
    >>> s1 = sphere()
    >>> s1.material.color = color(0.8,1.0,0.6)
    >>> s1.material.diffuse = 0.7
    >>> s1.material.specular = 0.2
    >>> s2 = sphere()
    >>> s2.transform = scaling(0.5,0.5,0.5)
    >>> w = default_world()
    >>> w.lights[0].position.compare(light.position)
    True

    >>> w.lights[0].intensity == light.intensity
    array([ True,  True,  True])
    >>> len(w.contains) == 2
    True
    >>> w.contains[0].material.color == color(0.8,1.0,0.6)
    array([ True,  True,  True])
    >>> w.contains[0].material.diffuse == 0.7 and w.contains[0].material.specular == 0.2
    True
    >>> w.contains[1].transform.compare(scaling(0.5,0.5,0.5))
    True

    """
    light = point_light(point(-10, 10, -10), color(1,1,1))
    s1 = shapes.sphere()
    s1.material.color = color(0.8,1.0,0.6)
    s1.material.diffuse = 0.7
    s1.material.specular = 0.2
    s2 = shapes.sphere()
    s2.transform = scaling(0.5,0.5,0.5)
    return World([light], [s1,s2])

def intersect_world(world, r):
    """
    >>> w = default_world()
    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> xs = intersect_world(w, r)
    >>> len(xs) == 4 and xs[0].t == 4 and xs[1].t == 4.5 and xs[2].t == 5.5 and xs[3].t == 6
    True
    """
    xs = []
    for obj in world.contains:
        xs.extend(shapes.intersect(obj, r))
    return shapes.intersections(*xs)

class Computations(object):
    def __init__(self, t, obj, point, eyev, normalv, reflectv, inside):
        self.t = t
        self.object = obj
        self.point = point
        self.eyev = eyev
        self.normalv = normalv
        self.inside = inside
        self.reflectv = reflectv
        self.over_point = None
        self.under_point = None
        self.n1 = None
        self.n2 = None

    def __repr__(self):
        return "Computation: {} ({}) {} {} {} {} {} {} {} {} {}".format(self.t, self.object, self.point, self.eyev, self.normalv, self.inside, self.reflectv, self.over_point, self.under_point, self.n1, self.n2)

def prepare_computations(intersection, r, xs):
    """
    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> shape = sphere()
    >>> i = intersection(4, shape)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i, r, xs)
    >>> comps.t == i.t
    True
    >>> comps.object == i.object
    True
    >>> comps.point.compare(point(0,0,-1))
    True

    >>> comps.eyev.compare(vector(0,0,-1))
    True

    >>> comps.normalv.compare(vector(0,0,-1))
    True

    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> shape = sphere()
    >>> i = intersection(4, shape)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i, r, xs)
    >>> comps.inside
    False

    >>> r = ray(point(0,0,0), vector(0,0,1))
    >>> shape = sphere()
    >>> i = intersection(1, shape)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i, r, xs)
    >>> comps.inside
    True
    >>> comps.point.compare(point(0,0,1))
    True

    >>> comps.eyev.compare(vector(0,0,-1))
    True

    >>> comps.normalv.compare(vector(0,0,-1))
    True

    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> shape = sphere()
    >>> shape.transform = translation(0,0,1)
    >>> i = intersection(5, shape)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i, r, xs)
    >>> comps.over_point[2] < -EPSILON/2
    True
    >>> comps.point[2] > comps.over_point[2]
    True

    >>> shape = plane()
    >>> r = ray(point(0,1,-1), vector(0,-np.sqrt(2)/2,np.sqrt(2)/2))
    >>> i = intersection(np.sqrt(2), shape)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i, r, xs)
    >>> comps.reflectv.compare(vector(0, np.sqrt(2)/2, np.sqrt(2)/2))
    True

    Refractive Index tests:
    >>> A = glass_sphere()
    >>> A.transform = scaling(2,2,2)
    >>> A.material.refractive_index = 1.5
    >>> B = glass_sphere()
    >>> B.transform = translation(0,0,-0.25)
    >>> B.material.refractive_index = 2.0
    >>> C = glass_sphere()
    >>> C.transform = translation(0,0,0.25)
    >>> C.material.refractive_index = 2.5

    Refractive Index tests:
    >>> r = ray(point(0,0,-4), vector(0,0,1))
    >>> xs = intersections(intersection(2,A), intersection(2.75,B), intersection(3.25,C), intersection(4.75,B), intersection(5.25,C), intersection(6,A))
    >>> comps = prepare_computations(xs[0], r, xs)
    >>> comps.n1 == 1.0 and comps.n2 == 1.5
    True
    >>> comps = prepare_computations(xs[1], r, xs)
    >>> comps.n1 == 1.5 and comps.n2 == 2.0
    True
    >>> comps = prepare_computations(xs[2], r, xs)
    >>> comps.n1 == 2.0 and comps.n2 == 2.5
    True
    >>> comps = prepare_computations(xs[3], r, xs)
    >>> comps.n1 == 2.5 and comps.n2 == 2.5
    True
    >>> comps = prepare_computations(xs[4], r, xs)
    >>> comps.n1 == 2.5 and comps.n2 == 1.5
    True
    >>> comps = prepare_computations(xs[5], r, xs)
    >>> comps.n1 == 1.5 and comps.n2 == 1.0
    True

    Underpoint:
    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> shape = glass_sphere()
    >>> shape.transform = translation(0,0,1)
    >>> i = intersection(5, shape)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i, r, xs)
    >>> comps.under_point[2] > EPSILON/2
    True
    >>> comps.point[2] < comps.under_point[2]
    True
    """
    p = shapes.position(r, intersection.t)
    n = normal_at(intersection.object, p, intersection)

    c = Computations(intersection.t,
                     intersection.object,
                     p,
                     -r.direction,
                     n,
                     reflect(r.direction, n),
                     False)

    if dot(c.normalv, c.eyev) < 0:
        c.inside = True
        c.normalv = -c.normalv

    c.over_point = c.point + c.normalv * EPSILON
    c.under_point = c.point - c.normalv * EPSILON

    containers = []
    for i in xs:
        if i == intersection:
            if len(containers) == 0:
                c.n1 = 1.0
            else:
                c.n1 = containers[-1].material.refractive_index

        if i.object in containers:
            containers.remove(i.object)
        else:
            containers.append(i.object)

        if i == intersection:
            if len(containers) == 0:
                c.n2 = 1.0
            else:
                c.n2 = containers[-1].material.refractive_index
            break

    return c

def schlick(comps):
    """
    >>> shape = glass_sphere()
    >>> r = ray(point(0,0,np.sqrt(2)/2), vector(0,1,0))
    >>> xs = intersections(intersection(-np.sqrt(2)/2, shape), intersection(np.sqrt(2)/2, shape))
    >>> comps = prepare_computations(xs[1], r, xs)
    >>> reflectance = schlick(comps)
    >>> reflectance == 1.0
    True

    >>> shape = glass_sphere()
    >>> r = ray(point(0,0,0), vector(0,1,0))
    >>> xs = intersections(intersection(-1, shape), intersection(1, shape))
    >>> comps = prepare_computations(xs[1], r, xs)
    >>> reflectance = schlick(comps)
    >>> np.isclose(reflectance, 0.04)
    True

    >>> shape = glass_sphere()
    >>> r = ray(point(0,0.99,-2), vector(0,0,1))
    >>> xs = intersections(intersection(1.8589, shape))
    >>> comps = prepare_computations(xs[0], r, xs)
    >>> reflectance = schlick(comps)
    >>> np.isclose(reflectance, 0.48873)
    True
    """
    cos = dot(comps.eyev, comps.normalv)
    if comps.n1 > comps.n2:
        n = comps.n1 / comps.n2
        sin2_t = n ** 2 * (1.0 - cos ** 2)
        if sin2_t > 1.0:
            return 1.0

        cos_t = np.sqrt(1.0 - sin2_t)
        cos = cos_t

    r0 = ((comps.n1 - comps.n2) / (comps.n1 + comps.n2)) ** 2
    return r0 + (1.0 - r0) * (1.0 - cos) ** 5

def refracted_color(world, comps, remaining=5, ilight=0):
    """
    >>> w = default_world()
    >>> shape = w.contains[0]
    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> xs = intersections(intersection(4,shape), intersection(6,shape))
    >>> comps = prepare_computations(xs[0], r, xs)
    >>> c = refracted_color(w, comps, 5)
    >>> c == color(0,0,0)
    array([ True,  True,  True])

    >>> w = default_world()
    >>> shape = w.contains[0]
    >>> shape.material.transparency = 1.0
    >>> shape.material.refractive_index = 1.5
    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> xs = intersections(intersection(4,shape), intersection(6,shape))
    >>> comps = prepare_computations(xs[0], r, xs)
    >>> c = refracted_color(w, comps, 0)
    >>> c == color(0,0,0)
    array([ True,  True,  True])

    >>> w = default_world()
    >>> shape = w.contains[0]
    >>> shape.material.transparency = 1.0
    >>> shape.material.refractive_index = 1.5
    >>> r = ray(point(0,0,np.sqrt(2)/2), vector(0,1,0))
    >>> xs = intersections(intersection(-np.sqrt(2),shape), intersection(np.sqrt(2),shape))
    >>> comps = prepare_computations(xs[1], r, xs)
    >>> c = refracted_color(w, comps, 5)
    >>> c == color(0,0,0)
    array([ True,  True,  True])

    >>> w = default_world()
    >>> A = w.contains[0]
    >>> A.material.ambient = 1.0
    >>> A.material.pattern = test_pattern()
    >>> B = w.contains[1]
    >>> B.material.transparency = 1.0
    >>> B.material.refractive_index = 1.5
    >>> r = ray(point(0,0,0.1), vector(0,1,0))
    >>> xs = intersections(intersection(-0.9899, A), intersection(-0.4899, B), intersection(0.4899, B), intersection(0.9899, A))
    >>> comps = prepare_computations(xs[2], r, xs)
    >>> c = refracted_color(w, comps, 5)
    >>> c
    array([0.        , 0.99888367, 0.04721668])

    Below is how the above test is written in the book
    np.isclose(c, color(0, 0.99888, 0.04725))
    array([ True,  True,  True])
    """
    if comps.object.material.transparency == 0 or remaining == 0:
        return color(0,0,0)

    n_ratio = comps.n1 / comps.n2
    cos_i = dot(comps.eyev, comps.normalv)
    sin2_t = n_ratio ** 2 * (1 - cos_i ** 2)

    if sin2_t > 1.0:
        return color(0,0,0)

    cos_t = np.sqrt(1.0 - sin2_t)
    direction = comps.normalv * (n_ratio * cos_i - cos_t) - \
                comps.eyev * n_ratio

    refracted_ray = shapes.ray(comps.under_point, direction)
    c = color_at(world, refracted_ray, remaining - 1) * \
        comps.object.material.transparency
    return c

def reflected_color(world, comps, remaining=5, ilight=0):
    """
    >>> w = default_world()
    >>> r = ray(point(0,0,0), vector(0,0,1))
    >>> shape = w.contains[1]
    >>> shape.material.ambient = 1
    >>> i = intersection(1, shape)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i, r, xs)
    >>> c = reflected_color(w, comps)
    >>> np.isclose(c, color(0,0,0))
    array([ True,  True,  True])

    >>> w = default_world()
    >>> shape = plane()
    >>> shape.material.reflective = 0.5
    >>> shape.transform = translation(0, -1, 0)
    >>> w.contains.append(shape)
    >>> r = ray(point(0,0,-3), vector(0,-np.sqrt(2)/2, np.sqrt(2)/2))
    >>> i = intersection(np.sqrt(2), shape)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i, r, xs)
    >>> c = reflected_color(w, comps)
    >>> w.contains = w.contains[:-1]
    >>> len(w.contains) == 2
    True
    >>> np.isclose(c, color(0.19033077, 0.23791346, 0.14274808))
    array([ True,  True,  True])

    >>> w = default_world()
    >>> shape = plane()
    >>> shape.material.reflective = 0.5
    >>> shape.transform = translation(0, -1, 0)
    >>> w.contains.append(shape)
    >>> r = ray(point(0,0,-3), vector(0,-np.sqrt(2)/2, np.sqrt(2)/2))
    >>> i = intersection(np.sqrt(2), shape)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i, r, xs)
    >>> c = reflected_color(w, comps, 0)
    >>> w.contains = w.contains[:-1]
    >>> len(w.contains) == 2
    True
    >>> np.isclose(c, color(0,0,0))
    array([ True,  True,  True])
    """
    if remaining == 0 or comps.object.material.reflective == 0:
        return color(0,0,0)

    reflect_ray = shapes.ray(comps.over_point, comps.reflectv)
    c = color_at(world, reflect_ray, remaining - 1, ilight)

    return c * comps.object.material.reflective

# TODO ilight is currently ignored
def shade_hit(world, comps, remaining=5, ilight=0):
    """
    >>> w = default_world()
    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> shape = w.contains[0]
    >>> i = intersection(4, shape)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i, r, xs)
    >>> c = shade_hit(w, comps)
    >>> np.isclose(c,color(0.38066, 0.47583, 0.28549589))
    array([ True,  True,  True])

    >>> w = default_world()
    >>> w.lights[0] = point_light(point(0,0.25,0), color(1,1,1))
    >>> r = ray(point(0,0,0), vector(0,0,1))
    >>> shape = w.contains[1]
    >>> i = intersection(0.5, shape)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i,r, xs)
    >>> c = shade_hit(w, comps)
    >>> np.isclose(c,color(0.90498, 0.90498, 0.90498))
    array([ True,  True,  True])

    >>> w = world()
    >>> w.lights = [point_light(point(0,0,-1), color(1,1,1))]
    >>> s1 = sphere()
    >>> s2 = sphere()
    >>> s2.transform = translation(0,0,10)
    >>> w.contains = [s1,s2]
    >>> r = ray(point(0,0,5), vector(0,0,1))
    >>> i = intersection(4, s2)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i, r, xs)
    >>> c = shade_hit(w, comps)
    >>> w.contains = []
    >>> w.lights = None
    >>> np.isclose(c, color(0.1,0.1,0.1))
    array([ True,  True,  True])

    >>> w = default_world()
    >>> shape = plane()
    >>> shape.material.reflective = 0.5
    >>> shape.transform = translation(0, -1, 0)
    >>> w.contains.append(shape)
    >>> r = ray(point(0,0,-3), vector(0,-np.sqrt(2)/2, np.sqrt(2)/2))
    >>> i = intersection(np.sqrt(2), shape)
    >>> xs = intersections(i)
    >>> comps = prepare_computations(i, r, xs)
    >>> c = shade_hit(w, comps)
    >>> w.contains = w.contains[:-1]
    >>> len(w.contains) == 2
    True
    >>> np.isclose(c, color(0.87675616, 0.92433885, 0.82917347))
    array([ True,  True,  True])

    >>> w = default_world()
    >>> floor = plane()
    >>> floor.transform = translation(0,-1,0)
    >>> floor.material.transparency = 0.5
    >>> floor.material.refractive_index = 1.5
    >>> w.contains.append(floor)
    >>> ball = sphere()
    >>> ball.material.color = color(1,0,0)
    >>> ball.material.ambient = 0.5
    >>> ball.transform = translation(0, -3.5, -0.5)
    >>> w.contains.append(ball)
    >>> r = ray(point(0,0,-3), vector(0,-np.sqrt(2)/2, np.sqrt(2)/2))
    >>> xs = intersections(intersection(np.sqrt(2), floor))
    >>> comps = prepare_computations(xs[0], r, xs)
    >>> c = shade_hit(w, comps, 5)
    >>> w.contains = w.contains[:-2]
    >>> len(w.contains) == 2
    True
    >>> np.isclose(c, color(0.93642, 0.68642539, 0.68642539))
    array([ True,  True,  True])

    >>> w = default_world()
    >>> floor = plane()
    >>> floor.transform = translation(0,-1,0)
    >>> floor.material.transparency = 0.5
    >>> floor.material.reflective = 0.5
    >>> floor.material.refractive_index = 1.5
    >>> w.contains.append(floor)
    >>> ball = sphere()
    >>> ball.material.color = color(1,0,0)
    >>> ball.material.ambient = 0.5
    >>> ball.transform = translation(0, -3.5, -0.5)
    >>> w.contains.append(ball)
    >>> r = ray(point(0,0,-3), vector(0,-np.sqrt(2)/2, np.sqrt(2)/2))
    >>> xs = intersections(intersection(np.sqrt(2), floor))
    >>> comps = prepare_computations(xs[0], r, xs)
    >>> c = shade_hit(w, comps, 5)
    >>> w.contains = w.contains[:-2]
    >>> len(w.contains) == 2
    True
    >>> np.isclose(c, color(0.93391, 0.69643, 0.69243))
    array([ True,  True,  True])
    """
    acc = color(0,0,0)
    for i in range(len(world.lights)):
        shadowed = is_shadowed(world, comps.over_point, i)
        surface = lighting(comps.object.material,
                           comps.object,
                           world.lights[i],
                           comps.over_point,
                           comps.eyev,
                           comps.normalv,
                           shadowed)
        reflected = reflected_color(world, comps, remaining, i)
        refracted = refracted_color(world, comps, remaining, i)

        mat = comps.object.material
        if mat.reflective > 0 and mat.transparency > 0:
            reflectance = schlick(comps)
            acc += surface + reflected * reflectance + refracted * (1.0 - reflectance)
        else:
            acc += surface + reflected + refracted
    return acc

def is_shadowed(world, point, ilight=0):
    """
    >>> w = default_world()
    >>> p = point(0,10,0)
    >>> is_shadowed(w, p)
    False

    >>> w = default_world()
    >>> p = point(10,-10,10)
    >>> is_shadowed(w, p)
    True

    >>> w = default_world()
    >>> p = point(-20,20,-20)
    >>> is_shadowed(w, p)
    False

    >>> w = default_world()
    >>> p = point(-2,2,-2)
    >>> is_shadowed(w, p)
    False

    """

    light = world.lights[ilight]
    v = light.position - point
    distance = magnitude(v)
    direction = normalize(v)
    r = shapes.ray(point, direction)
    intersections = intersect_world(world, r)
    intersections = [ i for i in intersections if i.object.material.casts_shadow ]
    h = shapes.hit(intersections)

    return h is not None and h.t < distance

def color_at(world, ray, remaining=5, ilight=0):
    """
    This method is called by render and prepares things for shade_hit

    >>> w = default_world()
    >>> r = ray(point(0,0,-5), vector(0,1,0))
    >>> c = color_at(w,r)
    >>> np.isclose(c, color(0,0,0))
    array([ True,  True,  True])

    >>> w = default_world()
    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> c = color_at(w,r)
    >>> np.isclose(c, color(0.38066, 0.47583, 0.28549589))
    array([ True,  True,  True])

    >>> w = default_world()
    >>> outer = w.contains[0]
    >>> outer.material.ambient = 1.0
    >>> inner = w.contains[1]
    >>> inner.material.ambient = 1.0
    >>> r = ray(point(0,0,0.75), vector(0,0,-1))
    >>> c = color_at(w,r)
    >>> c == inner.material.color
    array([ True,  True,  True])

    >>> w1 = world()
    >>> w1.lights = [point_light(point(0,0,0), color(1,1,1))]
    >>> lower = plane()
    >>> lower.material.reflective = 1
    >>> lower.transform = translation(0,-1,0)
    >>> upper = plane()
    >>> upper.material.reflective = 1
    >>> upper.transform = translation(0,1,0)
    >>> w1.contains.extend([lower, upper])
    >>> r = ray(point(0,0,0), vector(0,1,0))
    >>> c = color_at(w,r)
    >>> w1.contains = []
    >>> w1.lights = None
    >>> w1.lights is None
    True
    >>> len(w1.contains) == 0
    True
    """
    xs = intersect_world(world, ray)
    i = shapes.hit(xs)
    if i is None:
        return color(0,0,0)
    comps = prepare_computations(i, ray, xs)
    return shade_hit(world, comps, remaining, ilight)

def view_transform(fr, to, up):
    """
    >>> fr = point(0,0,0)
    >>> to = point(0,0,-1)
    >>> up = vector(0,1,0)
    >>> t = view_transform(fr, to, up)
    >>> t.compare(matrix4x4identity())
    True

    >>> fr = point(0,0,0)
    >>> to = point(0,0,1)
    >>> up = vector(0,1,0)
    >>> t = view_transform(fr, to, up)
    >>> t.compare(scaling(-1,1,-1))
    True

    >>> fr = point(0,0,8)
    >>> to = point(0,0,0)
    >>> up = vector(0,1,0)
    >>> t = view_transform(fr, to, up)
    >>> t.compare(translation(0,0,-8))
    True

    >>> fr = point(1,3,2)
    >>> to = point(4,-2,8)
    >>> up = vector(1,1,0)
    >>> t = view_transform(fr, to, up)
    >>> t
    array([[-5.07092553e-01,  5.07092553e-01,  6.76123404e-01,
            -2.36643191e+00],
           [ 7.67715934e-01,  6.06091527e-01,  1.21218305e-01,
            -2.82842712e+00],
           [-3.58568583e-01,  5.97614305e-01, -7.17137166e-01,
             4.44089210e-16],
           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             1.00000000e+00]])

    """
    forward = normalize(to - fr)
    upn = normalize(up)
    left = cross(forward, upn)
    true_up = cross(left, forward)
    orientation = matrix(left[0], left[1], left[2], 0,
                         true_up[0], true_up[1], true_up[2], 0,
                         -forward[0], -forward[1], -forward[2], 0,
                         0,0,0,1)
    return orientation * translation(-fr[0], -fr[1], -fr[2])

def view_transform_from_yaml(fr, to, up):
    fr = point(*fr)
    to = point(*to)
    up = vector(*up)
    return view_transform(fr, to, up)

class Camera(object):
    def __init__(self, hsize, vsize, field_of_view, xform=matrix4x4identity()):
        self.hsize = hsize
        self.vsize = vsize
        self.field_of_view = float(field_of_view)
        self.transform = xform
        self.half_width, self.half_height, self.pixel_size = self._compute_sizes()

    @classmethod
    def from_yaml(cls, obj) -> 'Camera':
        xform = view_transform_from_yaml(obj['from'], obj['to'], obj['up'])
        return cls(hsize=obj['width'],
                   vsize=obj['height'],
                   field_of_view=obj['field-of-view'],
                   xform=xform)

    def _compute_sizes(self):
        half_view = np.tan(self.field_of_view / 2)
        aspect = float(self.hsize) / float(self.vsize)
        if aspect >= 1:
            half_width = half_view
            half_height = half_view / aspect
        else:
            half_width = half_view * aspect
            half_height = half_view
        return half_width, half_height, half_width * 2 / self.hsize

def camera(hsize, vsize, field_of_view):
    """
    >>> hsize = 160
    >>> vsize = 120
    >>> field_of_view = np.pi / 2
    >>> c = camera(hsize, vsize, field_of_view)
    >>> c.hsize == 160 and c.vsize == 120 and c.field_of_view == np.pi / 2
    True
    >>> c.transform.compare(matrix4x4identity())
    True

    >>> c = camera(200, 125, np.pi / 2)
    >>> np.isclose(c.pixel_size, 0.01)
    True

    >>> c = camera(125, 200, np.pi / 2)
    >>> np.isclose(c.pixel_size,0.01)
    True
    """
    return Camera(hsize, vsize, field_of_view)

def ray_for_pixel(cam, px, py):
    """
    >>> c = camera(201, 101, np.pi/2)
    >>> r = ray_for_pixel(c, 100, 50)
    >>> r.origin.compare(point(0,0,0))
    True

    >>> r.direction.compare(vector(0,0,-1))
    True

    >>> c = camera(201, 101, np.pi/2)
    >>> r = ray_for_pixel(c, 0, 0)
    >>> r.origin.compare(point(0,0,0))
    True

    >>> r.direction.compare(vector(0.66519, 0.33259, -0.66851))
    True

    >>> c = camera(201, 101, np.pi/2)
    >>> c.transform = matrix_multiply(rotation_y(np.pi/4), translation(0,-2,5))
    >>> r = ray_for_pixel(c, 100, 50)
    >>> r.origin.compare(point(0,2,-5))
    True

    >>> r.direction.compare(vector(np.sqrt(2)/2, 0, -np.sqrt(2)/2))
    True
    """
    xoffset = (px + 0.5) * cam.pixel_size
    yoffset = (py + 0.5) * cam.pixel_size
    world_x = cam.half_width - xoffset
    world_y = cam.half_height - yoffset

    pixel = inverse(cam.transform) * point(world_x, world_y, -1)
    origin = inverse(cam.transform) * point(0,0,0)
    direction = normalize(pixel - origin)

    return shapes.ray(origin, direction)

def render(cam, world):
    """
    >>> w = default_world()
    >>> c = camera(11, 11, np.pi/2)
    >>> fr = point(0,0,-5)
    >>> to = point(0,0,0)
    >>> up = vector(0,1,0)
    >>> c.transform = view_transform(fr, to, up)
    >>> image = render(c,w)
    >>> np.isclose(pixel_at(image, 5, 5), color(0.38066119, 0.47582649, 0.28549589))
    array([ True,  True,  True])
    """
    image = canvas(cam.hsize, cam.vsize)
    for y in range(cam.vsize):
        for x in range(cam.hsize):
            r = ray_for_pixel(cam, x, y)
            c = color_at(world, r)
            write_pixel(image, x, y, c)

    return image

# https://jonasteuwen.github.io/numpy/python/multiprocessing/2017/01/07/multiprocessing-numpy-array.html
# http://thousandfold.net/cz/2014/05/01/sharing-numpy-arrays-between-processes-using-multiprocessing-and-ctypes/
# Pool size 4 on raspberry pi 3b+
def render_multi_helper(args):
    cam, world, window_x, window_y = args

    buffer = np.zeros((BLOCK_SIZE, BLOCK_SIZE, 3))
    for idx_x in range(window_x, window_x + BLOCK_SIZE):
        for idx_y in range(window_y, window_y + BLOCK_SIZE):
            r = ray_for_pixel(cam, idx_x, idx_y)
            c = color_at(world, r)
            buffer[idx_x-window_x, idx_y-window_y] = c

    write_pixels(image, buffer, window_x, window_y, BLOCK_SIZE)

def render_multi(cam, world, num_threads=4):
    """
    >>> w = default_world()
    >>> c = camera(11, 11, np.pi/2)
    >>> fr = point(0,0,-5)
    >>> to = point(0,0,0)
    >>> up = vector(0,1,0)
    >>> c.transform = view_transform(fr, to, up)
    >>> image = render_multi(c,w)
    >>> np.isclose(pixel_at(image, 5, 5), color(0.38066119, 0.47582649, 0.28549589))
    array([ True,  True,  True])
    """
    global image
    image = canvas(cam.hsize, cam.vsize)
    window_idxs = [(cam, world, i, j) for i, j in
                   itertools.product(range(0, cam.hsize, BLOCK_SIZE),
                                     range(0, cam.vsize, BLOCK_SIZE))]

    with Pool(num_threads) as p:
        _ = p.map(render_multi_helper, window_idxs)

    return np.ctypeslib.as_array(image.shared_arr)
    #return image.np_arr

def normal_at(shape, world_point, hit):
    """
    >>> s = sphere()
    >>> n = normal_at(s, point(1,0,0))
    >>> n
    array([1., 0., 0., 0.])

    >>> s = sphere()
    >>> n = normal_at(s, point(0,1,0))
    >>> n
    array([0., 1., 0., 0.])

    >>> s = sphere()
    >>> n = normal_at(s, point(0,0,1))
    >>> n
    array([0., 0., 1., 0.])

    >>> s = sphere()
    >>> n = normal_at(s, point(np.sqrt(3)/3,np.sqrt(3)/3,np.sqrt(3)/3))
    >>> n.compare(vector(np.sqrt(3)/3,np.sqrt(3)/3,np.sqrt(3)/3))
    True

    >>> s = sphere()
    >>> n = normal_at(s, point(np.sqrt(3)/3,np.sqrt(3)/3,np.sqrt(3)/3))
    >>> n.compare(normalize(n))
    True

    >>> s = sphere()
    >>> s.transform = translation(0,1,0)
    >>> n = normal_at(s, point(0, 1.70711, -0.70711))
    >>> n.compare(vector(0, 0.70711, -0.70711))
    True

    >>> s = sphere()
    >>> m = matrix_multiply(scaling(1,0.5,1), rotation_z(np.pi / 5))
    >>> s.transform = m
    >>> n = normal_at(s, point(0, np.sqrt(2)/2, -np.sqrt(2)/2))
    >>> n.compare(vector(0, 0.97014, -0.242535625))
    True
    """
    return shape.normal_at(world_point, hit)

def reflect(inp, norm):
    """
    >>> v = vector(1,-1,0)
    >>> n = vector(0,1,0)
    >>> r = reflect(v,n)
    >>> r.compare(vector(1,1,0))
    True

    >>> v = vector(0,-1,0)
    >>> n = vector(np.sqrt(2)/2, np.sqrt(2)/2, 0)
    >>> r = reflect(v,n)
    >>> r.compare(vector(1,0,0))
    True
    """
    return inp - norm * 2 * dot(inp, norm)

class Light(object):
    def __init__(self, intensity):
        self.intensity = intensity

class PointLight(Light):
    def __init__(self, position, intensity):
        Light.__init__(self, intensity)
        self.position = position

    @classmethod
    def from_yaml(cls, obj) -> 'PointLight':
        return cls(position=point(*obj['at']), intensity=color(*obj['intensity']))

def point_light(position, intensity):
    """
    >>> i = color(1,1,1)
    >>> p = point(0,0,0)
    >>> light = point_light(p,i)
    >>> light.position.compare(p)
    True

    >>> light.intensity == i
    array([ True,  True,  True])

    """
    return PointLight(position, intensity)

black = color(0,0,0)
def lighting(material, shape, light, point, eyev, normalv, in_shadow=False):
    """
    >>> m = material()
    >>> shape = sphere()
    >>> pos = point(0,0,0)
    >>> eyev = vector(0,0,-1)
    >>> normalv = vector(0,0,-1)
    >>> light = point_light(point(0,0,-10), color(1,1,1))
    >>> result = lighting(m, shape, light, pos, eyev, normalv)
    >>> np.isclose(result, color(1.9,1.9,1.9))
    array([ True,  True,  True])

    >>> m = material()
    >>> pos = point(0,0,0)
    >>> eyev = vector(0,np.sqrt(2)/2,-np.sqrt(2)/2)
    >>> normalv = vector(0,0,-1)
    >>> light = point_light(point(0,0,-10), color(1,1,1))
    >>> result = lighting(m, shape, light, pos, eyev, normalv)
    >>> np.isclose(result, color(1.0,1.0,1.0))
    array([ True,  True,  True])

    >>> m = material()
    >>> pos = point(0,0,0)
    >>> eyev = vector(0,0,-1)
    >>> normalv = vector(0,0,-1)
    >>> light = point_light(point(0,10,-10), color(1,1,1))
    >>> result = lighting(m, shape, light, pos, eyev, normalv)
    >>> np.isclose(result, color(0.7364, 0.7364, 0.7364))
    array([ True,  True,  True])

    >>> m = material()
    >>> pos = point(0,0,0)
    >>> eyev = vector(0,-np.sqrt(2)/2,-np.sqrt(2)/2)
    >>> normalv = vector(0,0,-1)
    >>> light = point_light(point(0,10,-10), color(1,1,1))
    >>> result = lighting(m, shape, light, pos, eyev, normalv)
    >>> np.isclose(result, color(1.6364, 1.6364, 1.6364))
    array([ True,  True,  True])

    >>> m = material()
    >>> pos = point(0,0,0)
    >>> eyev = vector(0,0,-1)
    >>> normalv = vector(0,0,-1)
    >>> light = point_light(point(0,0,10), color(1,1,1))
    >>> result = lighting(m, shape, light, pos, eyev, normalv)
    >>> np.isclose(result, color(0.1, 0.1, 0.1))
    array([ True,  True,  True])

    >>> m = material()
    >>> pos = point(0,0,0)
    >>> eyev = vector(0,0,-1)
    >>> normalv = vector(0,0,-1)
    >>> light = point_light(point(0,0,-10), color(1,1,1))
    >>> in_shadow = True
    >>> result = lighting(m, shape, light, pos, eyev, normalv, in_shadow)
    >>> np.isclose(result, color(0.1, 0.1, 0.1))
    array([ True,  True,  True])

    >>> m = material()
    >>> shape = sphere()
    >>> eyev = vector(0,0,-1)
    >>> normalv = vector(0,0,-1)
    >>> light = point_light(point(0,0,-10), color(1,1,1))
    >>> in_shadow = False
    >>> m.pattern = stripe_pattern(color(1,1,1), color(0,0,0))
    >>> m.ambient = 1
    >>> m.diffuse = 0
    >>> m.specular = 0
    >>> c1 = lighting(m, shape, light, point(0.9,0,0), eyev, normalv, in_shadow)
    >>> c2 = lighting(m, shape, light, point(1.1,0,0), eyev, normalv, in_shadow)
    >>> np.isclose(c1, color(1,1,1))
    array([ True,  True,  True])

    >>> np.isclose(c2, color(0,0,0))
    array([ True,  True,  True])
    """
    if material.pattern is not None:
        pcolor = material.pattern.pattern_at_shape(shape, point)
    else:
        pcolor = material.color

    effective_color = pcolor * light.intensity
    ambient = effective_color * material.ambient
    if in_shadow:
        return ambient

    lightv = normalize(light.position - point)
    light_dot_normal = dot(lightv, normalv)
    if light_dot_normal < 0:
        diffuse = black
        specular = black
    else:
        diffuse = effective_color * material.diffuse * light_dot_normal
        reflectv = reflect(-lightv, normalv)
        reflect_dot_eye = dot(reflectv, eyev)
        if reflect_dot_eye <= 0:
            specular = black
        else:
            factor = np.power(reflect_dot_eye, material.shininess)
            specular = light.intensity * material.specular * factor
    return ambient + diffuse + specular