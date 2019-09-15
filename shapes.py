from canvas import *
from matrix import *
from renderer import *
from vector import *

from copy import deepcopy
from functools import lru_cache
EPSILON = 0.000001

class BoundingBox(object):
    def __init__(self, min=point(np.inf, np.inf, np.inf), max=point(-np.inf, -np.inf, -np.inf)):
        self.min = min
        self.max = max

    def add_point(self, pt):
        min_x = self.min[0]
        min_y = self.min[1]
        min_z = self.min[2]
        max_x = self.max[0]
        max_y = self.max[1]
        max_z = self.max[2]

        if pt[0] < self.min[0]:
            min_x = pt[0]
        if pt[1] < self.min[1]:
            min_y = pt[1]
        if pt[2] < self.min[2]:
            min_z = pt[2]

        if pt[0] > self.max[0]:
            max_x = pt[0]
        if pt[1] > self.max[1]:
            max_y = pt[1]
        if pt[2] > self.max[2]:
            max_z = pt[2]

        self.min = point(min_x, min_y, min_z)
        self.max = point(max_x, max_y, max_z)

    def add_box(self, other):
        if self is other:
            return
        self.add_point(other.min)
        self.add_point(other.max)

    def contains_point(self, pt):
        return self.min[0] <= pt[0] <= self.max[0]\
               and self.min[1] <= pt[1] <= self.max[1]\
               and self.min[2] <= pt[2] <= self.max[2]

    def contains_box(self, other):
        return self is other or (self.contains_point(other.min) and self.contains_point(other.max))

    @lru_cache(maxsize=CACHE_SIZE)
    def transform(self, matrix):
        p1 = self.min
        p2 = point(self.min[0], self.min[1], self.max[2])
        p3 = point(self.min[0], self.max[1], self.min[2])
        p4 = point(self.min[0], self.max[1], self.max[2])
        p5 = point(self.max[0], self.min[1], self.min[2])
        p6 = point(self.max[0], self.min[1], self.max[2])
        p7 = point(self.max[0], self.max[1], self.min[2])
        p8 = self.max

        new_bbox = BoundingBox()

        for p in [p1, p2, p3, p4, p5, p6, p7, p8]:
            new_bbox.add_point(matrix * p)

        return new_bbox

    def intersect(self, ray):
        return self.local_intersect(ray)

    def local_intersect(self, ray_local):
        xtmin, xtmax = self._check_axis(ray_local.origin[0], ray_local.direction[0], self.min[0], self.max[0])
        ytmin, ytmax = self._check_axis(ray_local.origin[1], ray_local.direction[1], self.min[1], self.max[1])
        ztmin, ztmax = self._check_axis(ray_local.origin[2], ray_local.direction[2], self.min[2], self.max[2])

        tmin = max(xtmin, ytmin, ztmin)
        tmax = min(xtmax, ytmax, ztmax)

        if tmin > tmax:
            return []
        return intersections(intersection(tmin, self), intersection(tmax, self))

    def _check_axis(self, origin, direction, min, max):
        tmin_numerator = min - origin
        tmax_numerator = max - origin

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

    def split_bounds(self):
        dx = abs(self.max[0] - self.min[0])
        dy = abs(self.max[1] - self.min[1])
        dz = abs(self.max[2] - self.min[2])
        greatest = max(dx,dy,dz)

        x0, y0, z0 = self.min[0], self.min[1], self.min[2]
        x1, y1, z1 = self.max[0], self.max[1], self.max[2]

        if greatest == dx:
            x0 = x1 = x0 + dx / 2.0
        elif greatest == dy:
            y0 = y1 = y0 + dy / 2.0
        else:
            z0 = z1 = z0 + dz / 2.0

        mid_min = point(x0, y0, z0)
        mid_max = point(x1, y1, z1)

        return BoundingBox(min=self.min,max=mid_max), BoundingBox(min=mid_min,max=self.max)

    def cube(self):
        pass

    def __repr__(self):
        return 'BoundingBox: {} {}'.format(self.min, self.max)

class Material(object):
    def __init__(self, color=color(1,1,1), ambient=0.1, diffuse=0.9, specular=0.9, shininess=200.0, reflective=0.0, transparency=0.0, refractive_index=1.0, casts_shadow=True, pattern=None):
        if ambient < 0 or diffuse < 0 or specular < 0 or shininess < 0:
            raise ValueError("Materials expect non-negative floating point values.")
        self.color = color
        self.ambient = np.float64(ambient)
        self.diffuse = np.float64(diffuse)
        self.specular = np.float64(specular)
        self.shininess = np.float64(shininess)
        self.pattern = pattern
        self.reflective = np.float64(reflective)
        self.transparency = np.float64(transparency)
        self.refractive_index = np.float64(refractive_index)
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
        pat = None

        if obj is not None:
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

            if 'shadow' in obj:
                casts_shadow = obj['shadow']

            if 'pattern' in obj:
                pat = Pattern.from_yaml(obj['pattern'])

        return Material(color=c,
                        diffuse=diffuse,
                        ambient=ambient,
                        specular=specular,
                        shininess=shininess,
                        reflective=reflective,
                        transparency=transparency,
                        refractive_index=refractive_index,
                        casts_shadow=casts_shadow,
                        pattern=pat)

    def __repr__(self):
        return "Material c: {} a: {} d: {} sp: {} sh: {} refl: {} trans: {} refr: {} shadow: {} Pattern: {}".format(
            self.color,
            self.ambient,
            self.diffuse,
            self.specular,
            self.shininess,
            self.reflective,
            self.transparency,
            self.refractive_index,
            self.casts_shadow,
            self.pattern)

class Ray(object):
    def __init__(self, o, d):
        self.origin = o
        self.direction = d

    def __repr__(self):
        return "Ray: {} {}".format(self.origin, self.direction)

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

    @lru_cache(maxsize=CACHE_SIZE)
    def bounds(self):
        return BoundingBox(point(-1,-1,-1), point(1,1,1))

    @lru_cache(maxsize=CACHE_SIZE)
    def parent_space_bounds(self):
        return self.bounds().transform(self.transform)

    def divide(self, threshold):
        return

    def __repr__(self):
        return "{}: ({}) {}".format(
            type(self).__name__,
            self.material.__repr__(),
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

        if 'op' in tree:
            op_obj = tree['op']
        else:
            op_obj = tree['operation']
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
        if len(self.bounds().intersect(r)) > 0:
            leftxs = intersect(self.left, r)
            rightxs = intersect(self.right, r)
            xs = intersections(*(leftxs + rightxs))
            return self.filter_intersections(xs)
        return []

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

    @lru_cache(maxsize=CACHE_SIZE)
    def bounds(self):
        box = BoundingBox()
        for c in [self.left, self.right]:
            cbox = c.parent_space_bounds()
            box.add_box(cbox)
        return box

    def divide(self, threshold):
        self.left.divide(threshold)
        self.right.divide(threshold)

    def __repr__(self):
        return "{}: ({}) Left:\n{}\nRight:\n{} ".format(
            type(self).__name__,
            self.material.__repr__(),
            self.left.__repr__(),
            self.right.__repr__())

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

class Group(Shape):
    def __init__(self, material, transform, children):
        Shape.__init__(self, material, transform)
        self.children = set(children)
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
        bbox = self.bounds()
        if True or len(bbox.intersect(ray_local)) > 0:
            xs = []
            for sh in self.children:
                xs.extend(sh.intersect(ray_local))
            return intersections(*xs)
        return []

    def add_child(self, shs):
        if self is shs:
            raise ValueError("Don't add a group to its own children collection.")
        if type(shs) == set or type(shs) == list:
            self.children.update(shs)
            for sh in shs:
                sh.parent = self
        else:
            self.children.add(shs)
            shs.parent = self

    @lru_cache(maxsize=CACHE_SIZE)
    def bounds(self):
        box = BoundingBox()
        for c in self.children:
            cbox = c.parent_space_bounds()
            box.add_box(cbox)
        return box

    def divide(self, threshold):
        if threshold < len(self.children):
            left, right = self._partition_children()
            if len(left) > 0:
                self._make_subgroup(left)
            if len(right) > 0:
                self._make_subgroup(right)

        for c in self.children:
            c.divide(threshold)

    def _make_subgroup(self, children):
        sub_group = Group(self.material, self.transform, children)
        self.children.difference_update(children)
        self.add_child(sub_group)

    def _partition_children(self):
        left_box, right_box = self.bounds().split_bounds()
        left_group = set()
        right_group = set()

        for c in self.children:
            sub_box = c.bounds()
            if left_box.contains_box(sub_box):
                left_group.add(c)
            elif right_box.contains_box(sub_box):
                right_group.add(c)
        return left_group, right_group

    def __repr__(self):
        c_strings = []
        for c in self.children:
            c_strings.append(c.__repr__())
        grouping_string = '{}\n' * len(c_strings)
        return 'Group: ({}) {} Children:\n{}'.format(self.material.__repr__(), self.transform, grouping_string.format(*c_strings))

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

    @lru_cache(maxsize=CACHE_SIZE)
    def bounds(self):
        return BoundingBox(point(-np.inf,0,-np.inf), point(np.inf,0,np.inf))

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

    @lru_cache(maxsize=CACHE_SIZE)
    def bounds(self):
        a = abs(self.minimum)
        b = abs(self.maximum)
        limit = max(a,b)

        return BoundingBox(point(-limit, self.minimum, -limit), point(limit, self.maximum, limit))


class Cylinder(Shape):
    def __init__(self, material, transform, minimum=-np.inf, maximum=np.inf, closed=False):
        Shape.__init__(self, material, transform)
        self.minimum = minimum
        self.maximum = maximum
        self.closed = closed

    @classmethod
    def from_yaml(cls, obj) -> 'Cylinder':
        b = True
        if 'closed' in obj:
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
            if self.minimum < y0 < self.maximum:
                xs.append(intersection(t0, self))

            y1 = ray_local.origin[1] + t1 * ray_local.direction[1]
            if self.minimum < y1 < self.maximum:
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

    @lru_cache(maxsize=CACHE_SIZE)
    def bounds(self):
        return BoundingBox(point(-1,self.minimum,-1), point(1,self.maximum,1))


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

    @lru_cache(maxsize=CACHE_SIZE)
    def bounds(self):
        b = BoundingBox()
        b.add_point(self.p1)
        b.add_point(self.p2)
        b.add_point(self.p3)

        return b

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

    @lru_cache(maxsize=CACHE_SIZE)
    def bounds(self):
        b = BoundingBox()
        b.add_point(self.p1)
        b.add_point(self.p2)
        b.add_point(self.p3)

        return b


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

def intersect(sh, ray):
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
    return sh.intersect(ray)

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

def world_to_object(sh, pt):
    return sh.world_to_object(pt)

def normal_to_world(sh, normal):
    return sh.normal_to_world(normal)

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



import noise

BLACK = color(0,0,0)
WHITE = color(1,1,1)

class Pattern(object):
    def __init__(self, color1=WHITE, color2=BLACK, transform=matrix4x4identity()):
        self.transform = transform
        self.a = color1
        self.b = color2

    @classmethod
    def from_yaml(cls, obj) -> 'Pattern':
        xform = matrix4x4identity()
        if 'transform' in obj:
            xform = Transform.from_yaml(obj['transform'])

        typ = obj['type']
        # base patterns
        if typ == 'checkers':
            color1 = color(*obj['colors'][0])
            color2 = color(*obj['colors'][1])
            return Checkers(color1=color1, color2=color2, transform=xform)
        elif typ == 'gradient':
            color1 = color(*obj['colors'][0])
            color2 = color(*obj['colors'][1])
            return Gradient(color1=color1, color2=color2, transform=xform)
        elif typ == 'radial-gradient':
            color1 = color(*obj['colors'][0])
            color2 = color(*obj['colors'][1])
            return RadialGradient(color1=color1, color2=color2, transform=xform)
        elif typ == 'rings':
            color1 = color(*obj['colors'][0])
            color2 = color(*obj['colors'][1])
            return Ring(color1=color1, color2=color2, transform=xform)
        elif typ == 'stripe':
            color1 = color(*obj['colors'][0])
            color2 = color(*obj['colors'][1])
            return Stripe(color1=color1, color2=color2, transform=xform)

        # nested patterns
        elif typ == 'blended':
            return BlendedPattern(pattern1=Pattern.from_yaml(obj['left']),
                                  pattern2=Pattern.from_yaml(obj['right']))
        elif typ == 'nested':
            return NestedPattern(pattern1=Pattern.from_yaml(obj['primary']),
                                 pattern2=Pattern.from_yaml(obj['left']),
                                 pattern3=Pattern.from_yaml(obj['right']))
        elif typ == 'perturbed':
            freq = 0.4
            scale_factor = 0.3
            octaves = 1
            if 'frequency' in obj:
                freq = np.float64(obj['frequency'])
            if 'scale-factor' in obj:
                scale_factor = np.float64(obj['scale-factor'])
            if 'octaves' in obj:
                octaves = obj['octaves']
            return PerturbedPattern(pattern1=Pattern.from_yaml(obj['primary']),
                                    frequency=freq,
                                    scale_factor=scale_factor,
                                    octaves=octaves)

        # uv mapped patterns
        elif typ == 'map':
            mapping = obj['mapping']
            if mapping == 'cube':
                return CubeMapPattern(uv_left=Pattern.uv_from_yaml(obj['left']),
                                      uv_front=Pattern.uv_from_yaml(obj['front']),
                                      uv_right=Pattern.uv_from_yaml(obj['right']),
                                      uv_back=Pattern.uv_from_yaml(obj['back']),
                                      uv_up=Pattern.uv_from_yaml(obj['up']),
                                      uv_down=Pattern.uv_from_yaml(obj['down']),
                                      uv_map=CubeUVMap)
            # TODO consider supporting both caps
            elif mapping == 'cylindrical':
                if 'uv_pattern' in obj:
                    uv_cap = Pattern.uv_from_yaml(obj['uv_pattern'])
                    uv_body = Pattern.uv_from_yaml(obj['uv_pattern'])
                else:
                    uv_cap = Pattern.uv_from_yaml(obj['top'])
                    uv_body = Pattern.uv_from_yaml(obj['front'])
                return CylinderMapPattern(uv_cap=uv_cap,
                                          uv_body=uv_body,
                                          uv_map=CylinderUVMap,
                                          transform=xform)
            elif mapping == 'planar':
                return TextureMapPattern(uv_pattern=Pattern.uv_from_yaml(obj['uv_pattern']),
                                         uv_map=PlaneUVMap,
                                         transform=xform)
            elif mapping == 'spherical':
                return TextureMapPattern(uv_pattern=Pattern.uv_from_yaml(obj['uv_pattern']),
                                         uv_map=SphereUVMap,
                                         transform=xform)
        raise ValueError('Unable to parse pattern type: {}'.format(typ))

    @classmethod
    def uv_from_yaml(cls, obj):
        typ = obj['type']
        if typ == 'checkers':
            # colors is a list
            color1 = color(*obj['colors'][0])
            color2 = color(*obj['colors'][1])
            return UVChecker(color1=color1, color2=color2, width=obj['width'], height=obj['height'])
        elif typ == 'align_check':
            colors = obj['colors'] # dict
            return UVAlignCheck(main_color=colors['main'],
                                upper_left=colors['ul'],
                                upper_right=colors['ur'],
                                bottom_left=colors['bl'],
                                bottom_right=colors['br'])
        elif typ == 'image':
            file_path = obj['file']
            return UVTexturePattern(tcanvas=TextureCanvas(file_path))

        raise ValueError('Unable to parse uv pattern type: {}'.format(typ))

    # testing impl
    def pattern_at(self, pt):
        return color(pt[0], pt[1], pt[2])

    def pattern_at_shape(self, sh, pt):
        """
        >>> shape = sphere()
        >>> shape.transform = scaling(2,2,2)
        >>> pattern = stripe_pattern(WHITE, BLACK)
        >>> c = pattern.pattern_at_shape(shape, point(1.5,0,0))
        >>> c == color(1,1,1)
        array([ True,  True,  True])

        >>> shape = sphere()
        >>> pattern = stripe_pattern(WHITE, BLACK)
        >>> pattern.transform = scaling(2,2,2)
        >>> c = pattern.pattern_at_shape(shape, point(1.5,0,0))
        >>> c == color(1,1,1)
        array([ True,  True,  True])

        >>> shape = sphere()
        >>> shape.transform = scaling(2,2,2)
        >>> pattern = stripe_pattern(WHITE, BLACK)
        >>> pattern.transform = translation(0.5,0,0)
        >>> c = pattern.pattern_at_shape(shape, point(1.5,0,0))
        >>> c == color(1,1,1)
        array([ True,  True,  True])

        >>> shape = sphere()
        >>> shape.transform = scaling(2,2,2)
        >>> pat = Pattern()
        >>> c = pat.pattern_at_shape(shape, point(2,3,4))
        >>> np.isclose(c, color(1,1.5,2))
        array([ True,  True,  True])

        >>> shape = sphere()
        >>> pat = Pattern()
        >>> pat.transform = scaling(2,2,2)
        >>> c = pat.pattern_at_shape(shape, point(2,3,4))
        >>> np.isclose(c, color(1,1.5,2))
        array([ True,  True,  True])

        >>> shape = sphere()
        >>> shape.transform = scaling(2,2,2)
        >>> pat = Pattern()
        >>> pat.transform = translation(0.5,1,1.5)
        >>> c = pat.pattern_at_shape(shape, point(2.5,3,3.5))
        >>> np.isclose(c, color(0.75,0.5,0.25))
        array([ True,  True,  True])

        """
        object_point = sh.world_to_object(pt)
        pattern_point = inverse(self.transform) * object_point
        return self.pattern_at(pattern_point)

    def _get_color_1(self):
        return self.a

    def _get_color_2(self):
        return self.b

    def _predicate_eval(self, pred):
        if pred:
            return self._get_color_1()
        return self._get_color_2()


#######################
#
# UV Maps for Abstract patterns
#
#######################
class SphereUVMap(object):
    @classmethod
    def uv_map(cls, object_point) -> 'tuple':
        #u = 0.5 + np.arctan2(object_point[2], object_point[0]) / (2 * np.pi)
        #v = 0.5 - np.arcsin(object_point[1]) / np.pi
        theta = np.arctan2(object_point[0], object_point[2]) # TODO swap x and y?
        vec = vector(object_point[0], object_point[1], object_point[2])
        radius = magnitude(vec)
        phi = np.arccos(object_point[1] / radius)
        raw_u = theta / (2 * np.pi)
        u = 1 - (raw_u + 0.5)
        v = 1 - phi / np.pi
        return u,v


# TODO figure out how to pick which of the three faces
# TODO using the cube method is not correct. It distorts near the caps
class CylinderUVMap(object):
    @classmethod
    def uv_map(cls, object_point) -> 'tuple':
        face = cls._face_from_point(object_point)
        if face == 0:
            return face, cls._uv_cylinder_body(object_point)
        elif face == 1:
            return face, cls._uv_cylinder_cap(object_point)
        raise ValueError('Unknown cylinder face: {}'.format(face))

    @classmethod
    def _face_from_point(cls, pt):
        radius = np.sqrt(pt[0] ** 2 + pt[2] ** 2)
        abs_y = abs(pt[1])
        coord = max(radius, abs_y)

        if coord == radius: # face
            return 0
        elif coord == abs_y: # top cap
            return 1
        elif coord == -abs_y: # bottom cap
            return 1
        return 0

    @classmethod
    def _uv_cylinder_body(cls, object_point):
        theta = np.arctan2(object_point[2], object_point[0])
        raw_u = theta / (2 * np.pi)
        u = 1 - (raw_u + 0.5)
        v = object_point[1] % 1
        return u,v

    @classmethod
    def _uv_cylinder_cap(cls, object_point) -> 'tuple':
        # TODO change me
        theta = np.arctan2(object_point[2], object_point[0])
        raw_u = theta / (2 * np.pi)
        u = 1 - (raw_u + 0.5)
        v = object_point[1] % 1
        return u,v


class PlaneUVMap(object):
    @classmethod
    def uv_map(cls, object_point) -> 'tuple':
        u = object_point[0] % 1
        v = object_point[2] % 1
        return u,v


class CubeUVMap(object):
    @classmethod
    def uv_map(cls, object_point) -> 'tuple':
        face = cls._face_from_point(object_point)
        if face == 0:
            return face, cls._uv_cube_right(object_point)
        elif face == 1:
            return face, cls._uv_cube_left(object_point)
        elif face == 2:
            return face, cls._uv_cube_up(object_point)
        elif face == 3:
            return face, cls._uv_cube_down(object_point)
        elif face == 4:
            return face, cls._uv_cube_front(object_point)
        elif face == 5:
            return face, cls._uv_cube_back(object_point)
        raise ValueError('Unknown cube face: {}'.format(face))


    @classmethod
    def _face_from_point(cls, pt):
        abs_x = abs(pt[0])
        abs_y = abs(pt[1])
        abs_z = abs(pt[2])
        coord = max(abs_x, abs_y, abs_z)

        if coord == pt[0]:
            return 0
        elif coord == -pt[0]:
            return 1
        elif coord == pt[1]:
            return 2
        elif coord == -pt[1]:
            return 3
        elif coord == pt[2]:
            return 4
        return 5

    @classmethod
    def _uv_cube_left(cls, pt):
        u = ((pt[2] + 1) % 2.0) / 2.0
        v = ((pt[1] + 1) % 2.0) / 2.0
        return u,v

    @classmethod
    def _uv_cube_right(cls, pt):
        u = ((1 - pt[2]) % 2.0) / 2.0
        v = ((pt[1] + 1) % 2.0) / 2.0
        return u,v

    @classmethod
    def _uv_cube_front(cls, pt):
        u = ((pt[0] + 1) % 2.0) / 2.0
        v = ((pt[1] + 1) % 2.0) / 2.0
        return u,v

    @classmethod
    def _uv_cube_back(cls, pt):
        u = ((1 - pt[0]) % 2.0) / 2.0
        v = ((pt[1] + 1) % 2.0) / 2.0
        return u,v

    @classmethod
    def _uv_cube_up(cls, pt):
        u = ((pt[0] + 1) % 2.0) / 2.0
        v = ((1 - pt[2]) % 2.0) / 2.0
        return u,v

    @classmethod
    def _uv_cube_down(cls, pt):
        u = ((pt[0] + 1) % 2.0) / 2.0
        v = ((pt[2] + 1) % 2.0) / 2.0
        return u,v


#######################
#
# Abstract patterns
#
#######################
class TextureMapPattern(Pattern):
    def __init__(self, uv_pattern, uv_map, transform):
        Pattern.__init__(self, uv_pattern.a, uv_pattern.b, transform)
        self.uv_pattern = uv_pattern
        self.uv_map = uv_map

    def pattern_at(self, pt):
        return self.uv_pattern.uv_pattern_at(*self.uv_map.uv_map(pt))


class CubeMapPattern(Pattern):
    def __init__(self, uv_left, uv_front, uv_right, uv_back, uv_up, uv_down, uv_map, transform):
        Pattern.__init__(self, WHITE, BLACK, transform)
        self.faces = [uv_right, uv_left, uv_up, uv_down, uv_front, uv_back] # order matters
        self.uv_map = uv_map

    def pattern_at(self, pt):
        face, (u, v) = self.uv_map.uv_map(pt)
        return self.faces[face].uv_pattern_at(u,v)


class CylinderMapPattern(Pattern):
    def __init__(self, uv_cap, uv_body, uv_map, transform):
        Pattern.__init__(self, WHITE, BLACK, transform)
        self.faces = [uv_body, uv_cap] # order matters
        self.uv_map = uv_map

    def pattern_at(self, pt):
        face, (u, v) = self.uv_map.uv_map(pt)
        return self.faces[face].uv_pattern_at(u,v)


class BlendedPattern(Pattern):
    def __init__(self, pattern1, pattern2):
        Pattern.__init__(self, pattern1.a, pattern1.b)
        self.p1 = pattern1
        self.p2 = pattern2

        # TODO should be a copy or pattern1...
        if pattern2 is None:
            self.p2 = pattern1

    def pattern_at_shape(self, sh, pt):
        p1 = self.p1.pattern_at_shape(sh, pt)
        p2 = self.p2.pattern_at_shape(sh, pt)
        return (p1 + p2) / 2


class NestedPattern(Pattern):
    def __init__(self, pattern1, pattern2, pattern3):
        Pattern.__init__(self, pattern1.a, pattern1.b)
        self.p1 = pattern1
        self.p2 = pattern2
        self.p3 = pattern3

        if pattern2 is None:
            self.p2 = pattern1
        if pattern3 is None:
            self.p3 = pattern1

    def pattern_at_shape(self, sh, pt):
        self.p1.a = self.p2.pattern_at_shape(sh, pt)
        self.p1.b = self.p3.pattern_at_shape(sh, pt)
        return self.p1.pattern_at_shape(sh, pt)


class PerturbedPattern(Pattern):
    """
    With a frequency of 0.4 scale_factor works well at 1/3 of the scale of the
    patterns. E.g. an x-axis gradient (from 0 to 1) scaled at 6,1,1 works well
    with a scale_factor of 2
    """
    def __init__(self, pattern1, frequency=0.4, scale_factor=0.3, octaves=1):
        Pattern.__init__(self, pattern1.a, pattern1.b)
        self.p1 = pattern1
        self.freq = frequency
        self.scale_factor = scale_factor
        self.octaves = octaves

    def _perturb(self, pt):
        x = pt[0] / self.freq
        y = pt[1] / self.freq
        z = pt[2] / self.freq

        new_x = pt[0] + noise.pnoise3(x, y, z, self.octaves) * self.scale_factor
        z += 1.0
        new_y = pt[1] + noise.pnoise3(x, y, z, self.octaves) * self.scale_factor
        z += 1.0
        new_z = pt[2] + noise.pnoise3(x, y, z, self.octaves) * self.scale_factor

        return point(new_x, new_y, new_z)

    def pattern_at_shape(self, sh, pt):
        return self.p1.pattern_at_shape(sh, self._perturb(pt))


#######################
#
# UV patterns
#
#######################
class UVChecker(Pattern):
    def __init__(self, color1=WHITE, color2=BLACK, width=2, height=2):
        Pattern.__init__(self, color1, color2)
        self.width = width
        self.height = height

    def uv_pattern_at(self, u, v):
        u2 = np.floor(u * self.width)
        v2 = np.floor(v * self.height)
        return self._predicate_eval((u2 + v2) % 2 == 0)


class UVAlignCheck(Pattern):
    def __init__(self, main_color, upper_left, upper_right, bottom_left, bottom_right):
        Pattern.__init__(self, WHITE, BLACK)
        self.main = main_color
        self.ul = upper_left
        self.ur = upper_right
        self.bl = bottom_left
        self.br = bottom_right

    def uv_pattern_at(self, u, v):
        # remember: v=0 at the bottom, v=1 at the top
        if v > 0.8:
            if u < 0.2:
                return self.ul
            if u > 0.8:
                return self.ur
        elif v < 0.2:
            if u < 0.2:
                return self.bl
            if u > 0.8:
                return self.br

        return self.main


class UVTexturePattern(Pattern):
    def __init__(self, tcanvas):
        Pattern.__init__(self, WHITE, BLACK)
        self.canvas = tcanvas

    def uv_pattern_at(self, u, v):
        v = 1 - v
        y = u * (self.canvas.width - 1)
        x = v * (self.canvas.height - 1)
        return self.canvas.pixel_at(np.round(x).astype('int'), np.round(y).astype('int'))


#######################
#
# Concrete patterns
#
#######################
class Checkers(Pattern):
    def __init__(self, color1=WHITE, color2=BLACK, transform=matrix4x4identity()):
        Pattern.__init__(self, color1, color2, transform)

    def pattern_at(self, pattern_point):
        new_point = np.floor(pattern_point.arr)
        s = new_point[0] + new_point[1] + new_point[2]
        return self._predicate_eval(np.isclose((s % 2), 0))


class Gradient(Pattern):
    def __init__(self, color1=WHITE, color2=BLACK, transform=matrix4x4identity()):
        Pattern.__init__(self, color1, color2, transform)

    def pattern_at(self, pattern_point):
        distance = self.b - self.a
        fraction = pattern_point[0] - np.floor(pattern_point[0])
        return self.a + distance * fraction


class RadialGradient(Pattern):
    def __init__(self, color1=WHITE, color2=BLACK, transform=matrix4x4identity()):
        Pattern.__init__(self, color1, color2, transform)

    def pattern_at(self, pattern_point):
        color_a = self._get_color_1()
        distance = self._get_color_2() - color_a
        mag = np.sqrt(pattern_point[0] ** 2 + pattern_point[2] ** 2)
        fraction = mag - np.floor(mag)
        return color_a + distance * fraction


class Ring(Pattern):
    def __init__(self, color1=WHITE, color2=BLACK, transform=matrix4x4identity()):
        Pattern.__init__(self, color1, color2, transform)

    def pattern_at(self, pattern_point):
        return self._predicate_eval(
            np.floor(np.sqrt(pattern_point[0] ** 2 + pattern_point[2] ** 2)) % 2 == 0)


class Stripe(Pattern):
    def __init__(self, color1=WHITE, color2=BLACK, transform=matrix4x4identity()):
        Pattern.__init__(self, color1, color2, transform)

    def pattern_at(self, pattern_point):
        return self._predicate_eval(np.floor(pattern_point[0]) % 2 == 0)


if __name__ == '__main__':
    pass
