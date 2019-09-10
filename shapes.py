from canvas import *
from matrix import *
from pattern import *
from renderer import *
from vector import *

from copy import deepcopy

class Material(object):
    def __init__(self, color=color(1,1,1), ambient=0.1, diffuse=0.9, specular=0.9, shininess=200.0, reflective=0.0, transparency=0.0, refractive_index=1.0, casts_shadow=True):
        if ambient < 0 or diffuse < 0 or specular < 0 or shininess < 0:
            raise ValueError("Materials expect non-negative floating point values.")
        self.color = color
        self.ambient = np.float64(ambient)
        self.diffuse = np.float64(diffuse)
        self.specular = np.float64(specular)
        self.shininess = np.float64(shininess)
        self.pattern = None
        self.reflective = reflective
        self.transparency = transparency
        self.refractive_index = refractive_index
        self.casts_shadow = casts_shadow

    @classmethod
    def from_yaml(cls, obj) -> 'Material':
        c = color(1,1,1)
        ambient = 0.1
        diffuse = 0.9
        specular = 0.9
        shininess = 200.0
        reflective = 0.0
        transparency = 0.0
        refractive_index = 1.0
        casts_shadow = True

        if 'color' in obj:
            c = color(*obj['color'])

        if 'diffuse' in obj:
            diffuse = float(obj['diffuse'])

        if 'ambient' in obj:
            ambient = float(obj['ambient'])

        if 'specular' in obj:
            specular = float(obj['specular'])

        if 'shininess' in obj:
            shininess = float(obj['shininess'])

        if 'reflective' in obj:
            reflective = float(obj['reflective'])

        if 'transparency' in obj:
            transparency = float(obj['transparency'])

        if 'refractive-index' in obj:
            refractive_index = float(obj['refractive-index'])

        if 'casts-shadow' in obj:
            casts_shadow = obj['casts-shadow']

        return Material(color=c,
                        diffuse=diffuse,
                        ambient=ambient,
                        specular=specular,
                        shininess=shininess,
                        reflective=reflective,
                        transparency=transparency,
                        refractive_index=refractive_index,
                        casts_shadow=casts_shadow)

    def __repr__(self):
        return "c: {} a: {} d: {} sp: {} sh: {} refl: {} trans: {} refr: {}".format(
            self.color,
            self.ambient,
            self.diffuse,
            self.specular,
            self.shininess,
            self.reflective,
            self.transparency,
            self.refractive_index)

class Ray(object):
    def __init__(self, o, d):
        self.origin = o
        self.direction = d

class Shape(object):
    def __init__(self, material=Material(), transform=matrix4x4identity()):
        self.transform = transform
        self.material = material
        self.parent = None

    @classmethod
    def _recursive_helper(cls, obj, defines) -> 'Shape':
        if 'transform' not in obj:
            obj['transform'] = []

        if obj["add"] == "sphere":
            return Sphere.from_yaml(obj)
        elif obj["add"] == "plane":
            return Plane.from_yaml(obj)
        elif obj["add"] == "cube":
            return Cube.from_yaml(obj)
        elif obj["add"] == "cone":
            return Cone.from_yaml(obj)
        elif obj["add"] == "cylinder":
            return Cylinder.from_yaml(obj)
        elif obj["add"] == "group":
            if "material" in obj:
                for child in obj["children"]:
                    if "material" not in child:
                        child["material"] = deepcopy(obj["material"])
            return Group.from_yaml(obj, defines)
        elif obj["add"] == "csg":
            if "material" in obj:
                if "material" not in obj["left"]:
                    obj["left"]["material"] = deepcopy(obj["material"])
                if "material" not in obj["right"]:
                    obj["right"]["material"] = deepcopy(obj["material"])
            return CSG.from_yaml(obj, defines)
        elif obj["add"] == "obj":
            from obj_parser import OBJParser
            return OBJParser.from_yaml(obj, defines)

        return None

    # TODO maybe this needs to be self == other
    def includes(self, other):
        return self is other

    def intersect(self, ray_original):
        ray = transform(ray_original, inverse(self.transform))
        return self.local_intersect(ray)

    def normal_at(self, world_point, hit=None):
        local_point = self.world_to_object(world_point)
        local_normal = self.local_normal_at(local_point, hit)
        world_normal = self.normal_to_world(local_normal)
        n = vector(world_normal[0], world_normal[1], world_normal[2]) # make sure normal[3] is 0
        return normalize(n)

    def world_to_object(self, pt):
        if self.parent is not None:
            pt = self.parent.world_to_object(pt)

        return inverse(self.transform) * pt

    def normal_to_world(self, normal):
        normal = transpose(inverse(self.transform)) * normal
        normal_1 = vector(normal[0], normal[1], normal[2]) # make sure normal[3] is 0
        normal = normalize(normal_1)

        if self.parent is not None:
            normal = self.parent.normal_to_world(normal)

        return normal

    def __repr__(self):
        return "Shape: (Material: {} {} {} {} {} {} {} {} {}), Transform: {}".format(
            self.material.color,
            self.material.ambient,
            self.material.diffuse,
            self.material.specular,
            self.material.shininess,
            self.material.pattern,
            self.material.reflective,
            self.material.transparency,
            self.material.refractive_index,
            self.transform)


class CSG(Shape):
    def __init__(self, material, transform, op, left, right):
        Shape.__init__(self, material, transform)
        self.op = op
        self.left = left
        self.right = right
        self.left.parent = self
        self.right.parent = self

    @classmethod
    def from_yaml(cls, tree, defines) -> 'CSG':
        mat = Material()
        xform = matrix4x4identity()
        left = None
        right = None
        op = None

        if 'material' in tree:
            mat = Material.from_yaml(tree['material'])

        if 'transform' in tree:
            xform = Transform.from_yaml(tree['transform'])

        left_obj = tree['left']
        if 'add' in left_obj:
            left = Shape._recursive_helper(left_obj, defines)
        else:
            raise ValueError('Malformed CSG yaml', tree)

        right_obj = tree['right']
        if 'add' in right_obj:
            right = Shape._recursive_helper(right_obj, defines)
        else:
            raise ValueError('Malformed CSG yaml', tree)

        op_obj = tree['op']
        if op_obj == 'difference':
            op = CSGDifference
        elif op_obj == 'intersection':
            op = CSGIntersect
        elif op_obj == 'union':
            op = CSGUnion
        else:
            raise ValueError('Malformed CSG yaml', tree)

        return cls(material=mat, transform=xform, op=op, left=left, right=right)

    def includes(self, other):
        return self.left.includes(other) or self.right.includes(other)

    def local_intersect(self, r):
        leftxs = intersect(self.left, r)
        rightxs = intersect(self.right, r)
        xs = intersections(*(leftxs + rightxs))
        return self.filter_intersections(xs)

    # assume xs is already filtered
    def filter_intersections(self, xs):
        inl = False
        inr = False
        result = []
        for i in xs:
            lhit = self.left.includes(i.object)
            if self.op.intersection_allowed(lhit, inl, inr):
                result.append(i)
            if lhit:
                inl = not inl
            else:
                inr = not inr

        return result

class CSGUnion(object):
    @classmethod
    def intersection_allowed(cls, lhit, inl, inr):
        return (lhit and not inr) or (not lhit and not inl)

class CSGIntersect(object):
    @classmethod
    def intersection_allowed(cls, lhit, inl, inr):
        return (lhit and inr) or (not lhit and inl)

class CSGDifference(object):
    @classmethod
    def intersection_allowed(cls, lhit, inl, inr):
        return (lhit and not inr) or (not lhit and inl)

# TODO implement TriangleMesh aggregate shape

class Group(Shape):
    def __init__(self, material, transform, children):
        Shape.__init__(self, material, transform)
        self.children = children
        for child in self.children:
            child.parent = self


    @classmethod
    def from_yaml(cls, tree, defines) -> 'Group':
        mat = Material()
        xform = matrix4x4identity()
        children = set()

        if 'material' in tree:
            mat = Material.from_yaml(tree['material'])

        if 'transform' in tree:
            xform = Transform.from_yaml(tree['transform'])

        for obj in tree['children']:
            if "add" in obj:
                children.add(Shape._recursive_helper(obj, defines))

        return cls(material=mat, transform=xform, children=children)

    def includes(self, other):
        return any([c.includes(other) for c in self.children])

    def local_intersect(self, ray_local):
        xs = []
        for shape in self.children:
            xs.extend(shape.intersect(ray_local))
        return intersections(*xs)

    def add_child(self, shapes):
        if self is shapes:
            raise ValueError("Don't add a group to its own children collection.")
        if type(shapes) == set or type(shapes) == list:
            self.children.update(shapes)
            for sh in shapes:
                sh.parent = self
        else:
            self.children.add(shapes)
            shapes.parent = self

    def __repr__(self):
        return "Group: (Children: {}) (Material: {} {} {} {} {} {} {} {} {}), Transform: {}".format(
            self.children,
            self.material.color,
            self.material.ambient,
            self.material.diffuse,
            self.material.specular,
            self.material.shininess,
            self.material.pattern,
            self.material.reflective,
            self.material.transparency,
            self.material.refractive_index,
            self.transform)

class Sphere(Shape):
    def __init__(self, material, transform):
        Shape.__init__(self, material, transform)
        self.origin = point(0,0,0)

    @classmethod
    def from_yaml(cls, obj) -> 'Sphere':
        return cls(material=Material.from_yaml(obj['material']), transform=Transform.from_yaml(obj['transform']))

    def local_intersect(self, ray_local):
        sphere_to_ray = ray_local.origin - self.origin
        a = dot(ray_local.direction, ray_local.direction)
        b = 2 * dot(ray_local.direction, sphere_to_ray)
        c = dot(sphere_to_ray, sphere_to_ray) - 1
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return []
        return intersections(Intersection((-b - np.sqrt(discriminant)) / (2 * a), self),
                             Intersection((-b + np.sqrt(discriminant)) / (2 * a), self))

    def local_normal_at(self, local_point, hit=None):
        return local_point - point(0,0,0)


class Plane(Shape):
    def __init__(self, material, transform):
        Shape.__init__(self, material, transform)
        self.normalv = vector(0,1,0)

    @classmethod
    def from_yaml(cls, obj) -> 'Plane':
        return cls(material=Material.from_yaml(obj['material']), transform=Transform.from_yaml(obj['transform']))

    def local_intersect(self, ray_local):
        if abs(ray_local.direction[1]) < EPSILON:
            return []

        t = -ray_local.origin[1] / ray_local.direction[1]
        return intersections(Intersection(t, self))

    def local_normal_at(self, local_point, hit=None):
        return self.normalv


class Cube(Shape):
    def __init__(self, material, transform):
        Shape.__init__(self, material, transform)

    @classmethod
    def from_yaml(cls, obj) -> 'Cube':
        return cls(material=Material.from_yaml(obj['material']), transform=Transform.from_yaml(obj['transform']))

    def local_intersect(self, ray_local):
        xtmin, xtmax = self._check_axis(ray_local.origin[0], ray_local.direction[0])
        ytmin, ytmax = self._check_axis(ray_local.origin[1], ray_local.direction[1])
        ztmin, ztmax = self._check_axis(ray_local.origin[2], ray_local.direction[2])

        tmin = max(xtmin, ytmin, ztmin)
        tmax = min(xtmax, ytmax, ztmax)

        if tmin > tmax:
            return []
        return intersections(intersection(tmin, self), intersection(tmax, self))

    def local_normal_at(self, local_point, hit=None):
        abs_x = abs(local_point[0])
        abs_y = abs(local_point[1])
        abs_z = abs(local_point[2])
        maxc = max(abs_x, abs_y, abs_z)

        if np.isclose(maxc,abs_x):
            return vector(local_point[0], 0, 0)
        elif np.isclose(maxc,abs_y):
            return vector(0, local_point[1], 0)

        return vector(0, 0, local_point[2])

    def _check_axis(self, origin, direction):
        tmin_numerator = -1 - origin
        tmax_numerator =  1 - origin

        if abs(direction) >= EPSILON:
            tmin = tmin_numerator / direction
            tmax = tmax_numerator / direction
        else:
            tmin = tmin_numerator * np.inf
            if tmin == np.nan:
                tmin = np.inf
                if tmin_numerator < 0:
                    tmin = -np.inf
            tmax = tmax_numerator * np.inf
            if tmax == np.nan:
                tmax = np.inf
                if tmax_numerator < 0:
                    tmax = -np.inf

        if tmin > tmax:
            return tmax, tmin
        return tmin, tmax

class Cone(Shape):
    def __init__(self, material, transform, minimum=-np.inf, maximum=np.inf, closed=False):
        Shape.__init__(self, material, transform)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.closed = closed

    @classmethod
    def from_yaml(cls, obj) -> 'Cone':
        b = obj['closed']
        return cls(material=Material.from_yaml(obj['material']),
                   transform=Transform.from_yaml(obj['transform']),
                   minimum=obj['min'],
                   maximum=obj['max'],
                   closed=b)

    def local_intersect(self, ray_local):
        xs = []
        a = ray_local.direction[0] ** 2 + \
            ray_local.direction[2] ** 2 - \
            ray_local.direction[1] ** 2
        b = 2 * ray_local.origin[0] * ray_local.direction[0] + \
            2 * ray_local.origin[2] * ray_local.direction[2] - \
            2 * ray_local.origin[1] * ray_local.direction[1]
        c = ray_local.origin[0] ** 2 + ray_local.origin[2] ** 2 - \
            ray_local.origin[1] ** 2

        if np.isclose(a, 0):
            if np.isclose(b, 0):
                pass # a and b are both zero, skip intersecting the conic
            xs.append(intersection(-c / (2 * b), self))
        else:
            disc = b ** 2 - 4 * a * c
            if disc < 0:
                return []

            discsqrt = np.sqrt(disc)
            t0 = (-b - discsqrt) / (2 * a)
            t1 = (-b + discsqrt) / (2 * a)
            if t0 > t1:
                tmp = t0
                t0 = t1
                t1 = tmp

            y0 = ray_local.origin[1] + t0 * ray_local.direction[1]
            if self.minimum < y0 and y0 < self.maximum:
                xs.append(intersection(t0, self))

            y1 = ray_local.origin[1] + t1 * ray_local.direction[1]
            if self.minimum < y1 and y1 < self.maximum:
                xs.append(intersection(t1, self))

        self._intersect_caps(ray_local, xs)

        return intersections(*xs)

    def local_normal_at(self, local_point, hit=None):
        dist = local_point[0] ** 2 + local_point[2] ** 2
        if dist < 1 and (self.maximum - EPSILON) < local_point[1]:
            return vector(0,1,0)
        elif dist < 1 and (self.minimum + EPSILON) > local_point[1]:
            return vector(0,-1,0)

        y = np.sqrt(dist)
        if local_point[0] > 0:
            y = -y

        return vector(local_point[0], y, local_point[2])

    def _check_cap(self, ray_local, t, y_local):
        x = ray_local.origin[0] + t * ray_local.direction[0]
        z = ray_local.origin[2] + t * ray_local.direction[2]
        sm = (x ** 2 + z ** 2)
        return not sm > abs(y_local)

    def _intersect_caps(self, ray_local, xs):
        if not self.closed or np.isclose(ray_local.direction[1], 0):
            return

        t = (self.minimum - ray_local.origin[1]) / ray_local.direction[1]
        if self._check_cap(ray_local, t, self.minimum):
            xs.append(intersection(t, self))

        t = (self.maximum - ray_local.origin[1]) / ray_local.direction[1]
        if self._check_cap(ray_local, t, self.maximum):
            xs.append(intersection(t, self))


class Cylinder(Shape):
    def __init__(self, material, transform, minimum=-np.inf, maximum=np.inf, closed=False):
        Shape.__init__(self, material, transform)
        self.minimum = minimum
        self.maximum = maximum
        self.closed = closed

    @classmethod
    def from_yaml(cls, obj) -> 'Cylinder':
        b = obj['closed']
        return cls(material=Material.from_yaml(obj['material']),
                   transform=Transform.from_yaml(obj['transform']),
                   minimum=obj['min'],
                   maximum=obj['max'],
                   closed=b)

    def local_intersect(self, ray_local):
        a = ray_local.direction[0] ** 2 + ray_local.direction[2] ** 2
        xs = []

        if not np.isclose(a, 0):
            b = 2 * ray_local.origin[0] * ray_local.direction[0] + \
                2 * ray_local.origin[2] * ray_local.direction[2]
            c = ray_local.origin[0] ** 2 + ray_local.origin[2] ** 2 - 1

            disc = b ** 2 - 4 * a * c
            if disc < 0:
                return []

            discsqrt = np.sqrt(disc)
            t0 = (-b - discsqrt) / (2 * a)
            t1 = (-b + discsqrt) / (2 * a)
            if t0 > t1:
                tmp = t0
                t0 = t1
                t1 = tmp

            y0 = ray_local.origin[1] + t0 * ray_local.direction[1]
            if self.minimum < y0 and y0 < self.maximum:
                xs.append(intersection(t0, self))

            y1 = ray_local.origin[1] + t1 * ray_local.direction[1]
            if self.minimum < y1 and y1 < self.maximum:
                xs.append(intersection(t1, self))

        self._intersect_caps(ray_local, xs)

        return intersections(*xs)

    def local_normal_at(self, local_point, hit=None):
        dist = local_point[0] ** 2 + local_point[2] ** 2
        if dist < 1 and (self.maximum - EPSILON) < local_point[1]:
            return vector(0,1,0)
        elif dist < 1 and (self.minimum + EPSILON) > local_point[1]:
            return vector(0,-1,0)

        return vector(local_point[0], 0, local_point[2])

    def _check_cap(self, ray_local, t):
        x = ray_local.origin[0] + t * ray_local.direction[0]
        z = ray_local.origin[2] + t * ray_local.direction[2]
        sm = (x ** 2 + z ** 2)
        return not sm > 1

    def _intersect_caps(self, ray_local, xs):
        if not self.closed or np.isclose(ray_local.direction[1], 0):
            return

        t = (self.minimum - ray_local.origin[1]) / ray_local.direction[1]
        if self._check_cap(ray_local, t):
            xs.append(intersection(t, self))

        t = (self.maximum - ray_local.origin[1]) / ray_local.direction[1]
        if self._check_cap(ray_local, t):
            xs.append(intersection(t, self))


class Triangle(Shape):
    def __init__(self, material, transform, p1, p2, p3):
        Shape.__init__(self, material, transform)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.e1 = p2 - p1
        self.e2 = p3 - p1
        self.normal = normalize(cross(self.e2, self.e1))

    def local_normal_at(self, local_point, hit=None):
        return self.normal

    def local_intersect(self, r):
        dir_cross_e2 = cross(r.direction, self.e2)
        det = dot(self.e1, dir_cross_e2)
        if abs(det) < EPSILON:
            return []

        f = 1.0 / det
        p1_to_origin = r.origin - self.p1
        u = f * dot(p1_to_origin, dir_cross_e2)
        if u < 0 or u > 1:
            return []

        origin_cross_e1 = cross(p1_to_origin, self.e1)
        v = f * dot(r.direction, origin_cross_e1)
        if v < 0 or (u + v) > 1:
            return []

        t = f * dot(self.e2, origin_cross_e1)
        return intersections(intersection(t, self))

    def __repr__(self):
        return '{} {} {} {} {} {}'.format(self.p1, self.p2, self.p3, self.e1, self.e2, self.normal)


class SmoothTriangle(Shape):
    def __init__(self, material, transform, p1, p2, p3, n1, n2, n3):
        Shape.__init__(self, material, transform)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.e1 = p2 - p1
        self.e2 = p3 - p1

    def local_normal_at(self, local_point, hit):
        return self.n2 * hit.u + self.n3 * hit.v +  self.n1 * (1 - hit.u - hit.v)

    def local_intersect(self, r):
        dir_cross_e2 = cross(r.direction, self.e2)
        det = dot(self.e1, dir_cross_e2)
        if abs(det) < EPSILON:
            return []

        f = 1.0 / det
        p1_to_origin = r.origin - self.p1
        u = f * dot(p1_to_origin, dir_cross_e2)
        if u < 0 or u > 1:
            return []

        origin_cross_e1 = cross(p1_to_origin, self.e1)
        v = f * dot(r.direction, origin_cross_e1)
        if v < 0 or (u + v) > 1:
            return []

        t = f * dot(self.e2, origin_cross_e1)
        return intersections(intersection_with_uv(t, self, u, v))


class Intersection(object):
    def __init__(self, t, obj, u=None, v=None):
        self.t = t
        self.object = obj
        self.u = u
        self.v = v


def triangle(p1,p2,p3):
    """
    >>> p1 = point(0,1,0)
    >>> p2 = point(-1,0,0)
    >>> p3 = point(1,0,0)
    >>> t = triangle(p1,p2,p3)
    >>> t.p1.compare(p1) and t.p2.compare(p2) and t.p3.compare(p3) and t.e1.compare(vector(-1,-1,0)) and t.e2.compare(vector(1,-1,0)) and t.normal.compare(vector(0,0,-1))
    True
    """
    return Triangle(Material(), matrix4x4identity(), p1, p2, p3)

def group():
    """
    >>> g = group()
    >>> g.transform.compare(matrix4x4identity()) and len(g.children) == 0
    True

    >>> s = test_shape()
    >>> s.parent is None
    True

    >>> g = group()
    >>> s = test_shape()
    >>> g.add_child(s)
    >>> len(g.children) == 1 and s in g.children and s.parent == g
    True

    >>> g = group()
    >>> r = ray(point(0,0,0), vector(0,0,1))
    >>> xs = g.local_intersect(r)
    >>> len(xs) == 0
    True

    >>> g = group()
    >>> s1 = sphere()
    >>> s2 = sphere()
    >>> s2.transform = translation(0,0,-3)
    >>> s3 = sphere()
    >>> s3.transform = translation(5,0,0)
    >>> g.add_child(s1)
    >>> g.add_child(s2)
    >>> g.add_child(s3)
    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> xs = g.local_intersect(r)
    >>> len(xs) == 4 and xs[0].object == s2 and xs[1].object == s2 and xs[2].object == s1 and xs[3].object == s1
    True

    >>> g = group()
    >>> g.transform = scaling(2,2,2)
    >>> s = sphere()
    >>> s.transform = translation(5,0,0)
    >>> g.add_child(s)
    >>> r = ray(point(10,0,-10), vector(0,0,1))
    >>> xs = intersect(g, r)
    >>> len(xs) == 2
    True

    >>> g1 = group()
    >>> g2 = group()
    >>> g1.transform = rotation_y(np.pi/2)
    >>> g2.transform = scaling(2,2,2)
    >>> g1.add_child(g2)
    >>> s = sphere()
    >>> s.transform = translation(5,0,0)
    >>> g2.add_child(s)
    >>> p = world_to_object(s, point(-2, 0, -10))
    >>> p.compare(point(0,0,-1))
    True

    >>> g1 = group()
    >>> g2 = group()
    >>> g1.transform = rotation_y(np.pi/2)
    >>> g2.transform = scaling(1,2,3)
    >>> g1.add_child(g2)
    >>> s = sphere()
    >>> s.transform = translation(5,0,0)
    >>> g2.add_child(s)
    >>> n = normal_to_world(s, vector(np.sqrt(3)/3,np.sqrt(3)/3,np.sqrt(3)/3))
    >>> isclose(n, vector(0.28571429,  0.42857143, -0.85714286))
    array([ True,  True,  True,  True])

    >>> g1 = group()
    >>> g2 = group()
    >>> g1.transform = rotation_y(np.pi/2)
    >>> g2.transform = scaling(1,2,3)
    >>> g1.add_child(g2)
    >>> s = sphere()
    >>> s.transform = translation(5,0,0)
    >>> g2.add_child(s)
    >>> n = s.normal_at(point(1.7321, 1.1547, -5.5774))
    >>> isclose(n, vector(0.28570368,  0.42854315, -0.85716053))
    array([ True,  True,  True,  True])
    """
    return Group()

def cone():
    """
    >>> shape = cone()
    >>> tups = [(point(0, 0, -5), vector(0, 0, 1), 5, 5),
    ...         (point(0, 0, -5), vector(1, 1, 1), 8.66025, 8.66025),
    ...         (point(1, 1, -5), vector(-0.5, -1, 1), 4.55006, 49.44994)]
    >>> ts = []
    >>> for tup in tups:
    ...     direction = normalize(tup[1])
    ...     r = ray(tup[0], direction)
    ...     xs = shape.local_intersect(r)
    ...     ts.append(all([len(xs) == 2, np.isclose(xs[0].t, tup[2]), np.isclose(xs[1].t, tup[3])]))
    >>> all(ts)
    True

    >>> shape = cone()
    >>> direction = normalize(vector(0,1,1))
    >>> r = ray(point(0,0,-1), direction)
    >>> xs = shape.local_intersect(r)
    >>> len(xs) == 1 and np.isclose(xs[0].t, 0.35355)
    True

    >>> shape = cone()
    >>> shape.minimum = -0.5
    >>> shape.maximum = 0.5
    >>> shape.closed = True
    >>> tups = [(point(0, 0, -5), vector(0, 1, 0), 0),
    ...         (point(0, 0, -0.25), vector(0, 1, 1), 2),
    ...         (point(0, 0, -0.25), vector(0, 1, 0), 4)]
    >>> ts = []
    >>> for tup in tups:
    ...     direction = normalize(tup[1])
    ...     r = ray(tup[0], direction)
    ...     xs = shape.local_intersect(r)
    ...     ts.append(len(xs) == tup[2])
    >>> all(ts)
    True

    >>> shape = cone()
    >>> tups = [(point(0, 0, 0), vector(0, 0, 0)),
    ...         (point(1, 1, 1), vector(1, -np.sqrt(2), 1)),
    ...         (point(-1, -1, 0), vector(-1, 1, 0))]
    >>> ts = []
    >>> for tup in tups:
    ...     n = shape.local_normal_at(tup[0])
    ...     ts.append(n.compare(tup[1]))
    >>> all(ts)
    True
    """
    return Cone()

def cylinder():
    """
    >>> cyl = cylinder()
    >>> tups = [(point(1, 0, 0), vector(0, 1, 0)),
    ...         (point(0, 0, 0), vector(0, 1, 0)),
    ...         (point(0, 0, -5), vector(1, 1, 1))]
    >>> ts = []
    >>> for tup in tups:
    ...     direction = normalize(tup[1])
    ...     r = ray(tup[0], direction)
    ...     xs = cyl.local_intersect(r)
    ...     ts.append(len(xs) == 0)
    >>> all(ts)
    True

    >>> cyl = cylinder()
    >>> tups = [(point(1, 0, -5), vector(0, 0, 1), 5, 5),
    ...         (point(0, 0, -5), vector(0, 0, 1), 4, 6),
    ...         (point(0.5, 0, -5), vector(0.1, 1, 1), 6.80798191702732, 7.088723439378861)]
    >>> ts = []
    >>> for tup in tups:
    ...     direction = normalize(tup[1])
    ...     r = ray(tup[0], direction)
    ...     xs = cyl.local_intersect(r)
    ...     ts.append(all([len(xs) == 2, np.isclose(xs[0].t, tup[2]), np.isclose(xs[1].t, tup[3])]))
    >>> all(ts)
    True

    >>> cyl = cylinder()
    >>> tups = [(point(1, 0, 0), vector(1, 0, 0)),
    ...         (point(0, 5, -1), vector(0, 0, -1)),
    ...         (point(0, -2, 1), vector(0, 0, 1)),
    ...         (point(-1, 1, 0), vector(-1, 0, 0))]
    >>> ts = []
    >>> for tup in tups:
    ...     n = cyl.local_normal_at(tup[0])
    ...     ts.append(n.compare(tup[1]))
    >>> all(ts)
    True

    >>> cyl = cylinder()
    >>> cyl.minimum == -np.inf and cyl.maximum == np.inf
    True

    >>> cyl = cylinder()
    >>> cyl.minimum = 1.0
    >>> cyl.maximum = 2.0
    >>> tups = [(point(0, 1.5, 0), vector(0.1, 1, 0), 0),
    ...         (point(0, 3, -5), vector(0, 0, 1), 0),
    ...         (point(0, 0, -5), vector(0, 0, 1), 0),
    ...         (point(0, 2, -5), vector(0, 0, 1), 0),
    ...         (point(0, 1, -5), vector(0, 0, 1), 0),
    ...         (point(0, 1.5, -2), vector(0, 0, 1), 2)]
    >>> ts = []
    >>> for tup in tups:
    ...     direction = normalize(tup[1])
    ...     r = ray(tup[0], direction)
    ...     xs = cyl.local_intersect(r)
    ...     ts.append(len(xs) == tup[2])
    >>> all(ts)
    True

    >>> cyl = cylinder()
    >>> cyl.closed
    False

    >>> cyl = cylinder()
    >>> cyl.minimum = 1.0
    >>> cyl.maximum = 2.0
    >>> cyl.closed = True
    >>> tups = [(point(0, 3, 0), vector(0, -1, 0), 2),
    ...         (point(0, 3, -2), vector(0, -1, 2), 2),
    ...         (point(0, 4, -2), vector(0, -1, 1), 2),
    ...         (point(0, 0, -2), vector(0, 1, 2), 2),
    ...         (point(0, -1, -2), vector(0, 1, 1), 2)]
    >>> ts = []
    >>> for tup in tups:
    ...     direction = normalize(tup[1])
    ...     r = ray(tup[0], direction)
    ...     xs = cyl.local_intersect(r)
    ...     ts.append(len(xs) == tup[2])
    >>> all(ts)
    True

    >>> cyl = cylinder()
    >>> cyl.minimum = 1.0
    >>> cyl.maximum = 2.0
    >>> cyl.closed = True
    >>> tups = [(point(0, 1, 0), vector(0, -1, 0)),
    ...         (point(0.5, 1, 0), vector(0, -1, 0)),
    ...         (point(0, 1, 0.5), vector(0, -1, 0)),
    ...         (point(0, 2, 0), vector(0, 1, 0)),
    ...         (point(0.5, 2, 0), vector(0, 1, 0)),
    ...         (point(0, 2, 0.5), vector(0, 1, 0))]
    >>> ts = []
    >>> for tup in tups:
    ...     n = cyl.local_normal_at(tup[0])
    ...     ts.append(n.compare(tup[1]))
    >>> all(ts)
    True
    """
    return Cylinder()

def cube():
    """
    >>> c = cube()
    >>> tups = [(point(5, 0.5, 0), vector(-1, 0, 0), 4, 6),
    ...         (point(-5, 0.5, 0), vector(1, 0, 0), 4, 6),
    ...         (point(0.5, 5, 0), vector(0, -1, 0), 4, 6),
    ...         (point(0.5, -5, 0), vector(0, 1, 0), 4, 6),
    ...         (point(0.5, 0, 5), vector(0, 0, -1), 4, 6),
    ...         (point(0.5, 0, -5), vector(0, 0, 1), 4, 6),
    ...         (point(0, 0.5, 0), vector(0, 0, 1), -1, 1)]
    >>> ts = []
    >>> for tup in tups:
    ...     r = ray(tup[0], tup[1])
    ...     xs = c.local_intersect(r)
    ...     ts.append(all([len(xs) == 2, xs[0].t == tup[2], xs[1].t == tup[3]]))
    >>> all(ts)
    True

    >>> c = cube()
    >>> tups = [(point(-2, 0, 0), vector(0.2673, 0.5345, 0.8018)),
    ...         (point(0, -2, 0), vector(0.8018, 0.2673, 0.5345)),
    ...         (point(0, 0, -2), vector(0.5345, 0.8018, 0.2673)),
    ...         (point(2, 0, 2), vector(0, 0, -1)),
    ...         (point(0, 2, 2), vector(0, -1, 0)),
    ...         (point(2, 2, 0), vector(-1, 0, 0))]
    >>> ts = []
    >>> for tup in tups:
    ...     r = ray(tup[0], tup[1])
    ...     xs = c.local_intersect(r)
    ...     ts.append(len(xs) == 0)
    >>> all(ts)
    True

    >>> c = cube()
    >>> tups = [(point(1, 0.5, -0.8), vector(1, 0, 0)),
    ...         (point(-1, -0.2, 0.9), vector(-1, 0, 0)),
    ...         (point(-0.4, 1, -0.1), vector(0, 1, 0)),
    ...         (point(0.3, -1, -0.7), vector(0, -1, 0)),
    ...         (point(-0.6, 0.3, 1), vector(0, 0, 1)),
    ...         (point(0.4, 0.4, -1), vector(0, 0, -1)),
    ...         (point(1, 1, 1), vector(1, 0, 0)),
    ...         (point(-1, -1, -1), vector(-1, 0, 0))]
    >>> ts = []
    >>> for tup in tups:
    ...     p = tup[0]
    ...     n = c.local_normal_at(p)
    ...     ts.append(n.compare(tup[1]))
    >>> all(ts)
    True
    """
    return Cube(Material(), matrix4x4identity())

def plane():
    """
    >>> p = plane()
    >>> n1 = p.local_normal_at(point(0,0,0))
    >>> n2 = p.local_normal_at(point(10,0,-10))
    >>> n3 = p.local_normal_at(point(-5,0,150))
    >>> n1.compare(vector(0,1,0))
    True
    >>> n2.compare(vector(0,1,0))
    True
    >>> n3.compare(vector(0,1,0))
    True

    >>> p = plane()
    >>> r = ray(point(0,10,0), vector(0,0,1))
    >>> xs = p.local_intersect(r)
    >>> len(xs) == 0
    True

    >>> p = plane()
    >>> r = ray(point(0,0,0), vector(0,0,1))
    >>> xs = p.local_intersect(r)
    >>> len(xs) == 0
    True

    >>> p = plane()
    >>> r = ray(point(0,1,0), vector(0,-1,0))
    >>> xs = p.local_intersect(r)
    >>> len(xs) == 1 and np.isclose(xs[0].t,1) and xs[0].object == p
    True

    >>> p = plane()
    >>> r = ray(point(0,-1,0), vector(0,1,0))
    >>> xs = p.local_intersect(r)
    >>> len(xs) == 1 and np.isclose(xs[0].t,1) and xs[0].object == p
    True
    """
    return Plane()

def ray(o, d):
    """
    >>> p = point(1,2,3)
    >>> d = vector(4,5,6)
    >>> r = ray(p,d)
    >>> r.origin.compare(p)
    True

    >>> r.direction.compare(d)
    True
    """
    return Ray(o,d)

def position(ray, t):
    """
    >>> r = ray(point(2,3,4), vector(1,0,0))
    >>> position(r, 0).compare(point(2,3,4))
    True

    >>> position(r, 1).compare(point(3,3,4))
    True

    >>> position(r,-1).compare(point(1,3,4))
    True

    >>> position(r,2.5).compare(point(4.5,3,4))
    True
    """
    return ray.origin + ray.direction * t

def sphere():
    """
    >>> s = sphere()
    >>> s.transform.compare(matrix4x4identity())
    True
    """
    return Sphere(material(), matrix4x4identity())

def glass_sphere():
    """
    >>> s = glass_sphere()
    >>> s.transform.compare(matrix4x4identity())
    True
    >>> s.material.transparency == 1.0
    True
    >>> s.material.refractive_index == 1.5
    True
    """
    s = Sphere()
    s.material.transparency = 1.0
    s.material.refractive_index = 1.5
    return s

def intersect(shape, ray):
    """
    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> s = sphere()
    >>> xs = intersect(s,r)
    >>> len(xs) == 2
    True
    >>> xs[0].t == 4.0 and xs[1].t == 6.0
    True

    >>> r = ray(point(0,1,-5), vector(0,0,1))
    >>> s = sphere()
    >>> xs = intersect(s,r)
    >>> len(xs) == 2
    True
    >>> xs[0].t == 5.0 and xs[1].t == 5.0
    True

    >>> r = ray(point(0,2,-5), vector(0,0,1))
    >>> s = sphere()
    >>> xs = intersect(s,r)
    >>> len(xs) == 0
    True

    >>> r = ray(point(0,0,0), vector(0,0,1))
    >>> s = sphere()
    >>> xs = intersect(s,r)
    >>> len(xs) == 2
    True
    >>> xs[0].t == -1.0 and xs[1].t == 1.0
    True

    >>> r = ray(point(0,0,5), vector(0,0,1))
    >>> s = sphere()
    >>> xs = intersect(s,r)
    >>> len(xs) == 2
    True
    >>> xs[0].t == -6.0 and xs[1].t == -4.0
    True

    >>> r = ray(point(0,0,5), vector(0,0,1))
    >>> s = sphere()
    >>> xs = intersect(s,r)
    >>> len(xs) == 2
    True
    >>> id(xs[0].object) == id(s) and id(xs[1].object) == id(s)
    True

    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> s = sphere()
    >>> s.transform = scaling(2,2,2)
    >>> xs = intersect(s,r)
    >>> len(xs) == 2 and xs[0].t == 3 and xs[1].t == 7
    True

    >>> r = ray(point(0,0,-5), vector(0,0,1))
    >>> s = sphere()
    >>> s.transform = translation(5,0,0)
    >>> xs = intersect(s,r)
    >>> len(xs) == 0
    True
    """
    return shape.intersect(ray)

def intersection_with_uv(t, obj, u, v):
    return Intersection(t, obj, u, v)

def intersection(t, obj):
    """
    >>> s = sphere()
    >>> i = intersection(3.5, s)
    >>> i.t == 3.5 and id(s) == id(i.object)
    True
    """
    return Intersection(t, obj)

def intersections(*args):
    """
    >>> s = sphere()
    >>> i1 = intersection(1,s)
    >>> i2 = intersection(2,s)
    >>> xs = intersections(i1,i2)
    >>> len(xs) == 2 and xs[0].t == 1 and xs[1].t == 2
    True
    """
    return sorted(list(args), key=lambda i: i.t)

def hit(intersections):
    """
    >>> s = sphere()
    >>> i1 = intersection(1,s)
    >>> i2 = intersection(2,s)
    >>> xs = intersections(i1,i2)
    >>> i = hit(xs)
    >>> i == i1
    True

    >>> s = sphere()
    >>> i1 = intersection(-1,s)
    >>> i2 = intersection(1,s)
    >>> xs = intersections(i1,i2)
    >>> i = hit(xs)
    >>> i == i2
    True

    >>> s = sphere()
    >>> i1 = intersection(-2,s)
    >>> i2 = intersection(-1,s)
    >>> xs = intersections(i1,i2)
    >>> i = hit(xs)
    >>> i is None
    True

    >>> s = sphere()
    >>> i1 = intersection(5,s)
    >>> i2 = intersection(7,s)
    >>> i3 = intersection(-3,s)
    >>> i4 = intersection(2,s)
    >>> xs = intersections(i1,i2,i3,i4)
    >>> i = hit(xs)
    >>> i == i4
    True
    """
    for i in intersections:
        if i.t > 0:
            return i
    return None

def transform(r, matrix):
    """
    >>> r = ray(point(1,2,3), vector(0,1,0))
    >>> m = translation(3,4,5)
    >>> r2 = transform(r,m)
    >>> r2.origin.compare(point(4,6,8))
    True

    >>> r2.direction.compare(vector(0,1,0))
    True

    >>> r = ray(point(1,2,3), vector(0,1,0))
    >>> m = scaling(2,3,4)
    >>> r2 = transform(r,m)
    >>> r2.origin.compare(point(2,6,12))
    True

    >>> r2.direction.compare(vector(0,3,0))
    True
    """
    return ray(matrix * r.origin,
               matrix * r.direction)

def world_to_object(shape, pt):
    return shape.world_to_object(pt)

def normal_to_world(shape, normal):
    return shape.normal_to_world(normal)

def test_shape():
    return Shape()

def material():
    """
    >>> m = material()
    >>> m.color == color(1,1,1)
    array([ True,  True,  True])

    >>> m.ambient == 0.1 and m.diffuse == 0.9 and m.specular == 0.9 and m.shininess == 200.0
    True

    >>> s = sphere()
    >>> sm = s.material
    >>> m = material()
    >>> sm.color == m.color
    array([ True,  True,  True])
    >>> sm.ambient == m.ambient and sm.diffuse == m.diffuse and sm.specular == m.specular and sm.shininess == m.shininess
    True

    >>> s = sphere()
    >>> m = material()
    >>> m.ambient = 1
    >>> s.material = m
    >>> s.material.ambient == 1
    True

    >>> m = material()
    >>> m.reflective == 0
    True

    >>> m.transparency == 0
    True
    >>> m.refractive_index == 1
    True
    """
    return Material(color(1,1,1),0.1,0.9,0.9,200.0)

if __name__ == '__main__':
    tups = [(CSGUnion, True, True, True, False),
            (CSGUnion, True, True, False, True),
            (CSGUnion, True, False, True, False),
            (CSGUnion, True, False, False, True),
            (CSGUnion, False, True, True, False),
            (CSGUnion, False, True, False, False),
            (CSGUnion, False, False, True, True),
            (CSGUnion, False, False, False, True),
            (CSGIntersect, True, True, True, True),
            (CSGIntersect, True, True, False, False),
            (CSGIntersect, True, False, True, True),
            (CSGIntersect, True, False, False, False),
            (CSGIntersect, False, True, True, True),
            (CSGIntersect, False, True, False, True),
            (CSGIntersect, False, False, True, False),
            (CSGIntersect, False, False, False, False),
            (CSGDifference, True, True, True, False),
            (CSGDifference, True, True, False, True),
            (CSGDifference, True, False, True, False),
            (CSGDifference, True, False, False, True),
            (CSGDifference, False, True, True, True),
            (CSGDifference, False, True, False, True),
            (CSGDifference, False, False, True, False),
            (CSGDifference, False, False, False, False)]
    for tup in tups:
        result = tup[0].intersection_allowed(tup[1], tup[2], tup[3])
        #print(result == tup[4])

    tups = [
        (CSGUnion, 0, 3),
        (CSGIntersect, 1, 2),
        (CSGDifference, 0, 1)
    ]
    s1 = sphere()
    s2 = cube()
    for tup in tups:
        c = CSG(Material(), matrix4x4identity(), tup[0], s1, s2)
        xs = intersections(intersection(1, s1),
                           intersection(2, s2),
                           intersection(3, s1),
                           intersection(4, s2))
        result = c.filter_intersections(xs)
        #print(len(result) == 2 and result[0] == xs[tup[1]] and result[1] == xs[tup[2]])

    c = CSG(Material(), matrix4x4identity(), CSGUnion, s1, s2)
    r = ray(point(0,2,-5), vector(0,0,1))
    xs = c.local_intersect(r)

    s2.transform = Transform.translate(0,0,0.5)
    c = CSG(Material(), matrix4x4identity(), CSGUnion, s1, s2)
    r = ray(point(0,0,-5), vector(0,0,1))
    xs = c.local_intersect(r)
    print(len(xs) == 2)
    print(xs[0].t == 4)
    print(xs[0].object == s1)
    print(xs[1].t == 6.5)
    print(xs[1].object == s2)


