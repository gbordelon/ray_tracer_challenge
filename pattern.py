from .canvas import color
from .shape import *
from .matrix import *

import noise

BLACK = color(0,0,0)
WHITE = color(1,1,1)

class Pattern(object):
    def __init__(self, color1=WHITE, color2=BLACK):
        self.transform = matrix4x4identity()
        self.a = color1
        self.b = color2

    # testing impl
    def pattern_at(self, pt):
        return color(pt[0], pt[1], pt[2])

    def pattern_at_shape(self, shape, pt):
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
        object_point = shape.world_to_object(pt)
        object_point = self._uv_map_hook(object_point)
        pattern_point = inverse(self.transform) * object_point
        return self.pattern_at(pattern_point)

    def _uv_map_hook(self, object_point):
        return object_point

    def _propagate_hook(self, hook):
        self._uv_map_hook = hook

    def _get_color_1(self):
        return self.a

    def _get_color_2(self):
        return self.b

    def _predicate_eval(self, pred):
        if pred:
            return self._get_color_1()
        return self._get_color_2()

class SphereUVMapPattern(Pattern):
    """
    this is a 2D point for a flat pattern but several
    patterns use x,z so the caller of this pattern
    needs to know to apply a rotate_x transform.
    """
    def __init__(self, pattern1):
        Pattern.__init__(self, pattern1.a, pattern1.b)
        self.p1 = pattern1
        self.p1._propagate_hook(self._uv_map_hook)

    def _uv_map_hook(self, object_point):
        # assume sphere so object point contains unit vector values
        u = 0.5 + np.arctan2(object_point[2], object_point[0]) / (2 * np.pi)
        v = 0.5 - np.arcsin(object_point[1]) / np.pi
        return point(u,v,0)

    def pattern_at_shape(self, shape, pt):
        return self.p1.pattern_at_shape(shape, pt)

class BlendedPattern(Pattern):
    def __init__(self, pattern1, pattern2):
        Pattern.__init__(self, pattern1.a, pattern1.b)
        self.p1 = pattern1
        self.p2 = pattern2

        # TODO should be a copy or pattern1...
        if pattern2 is None:
            self.p2 = pattern1

    def _propagate_hook(self, hook):
        self.p1._propagate_hook(hook)
        self.p2._propagate_hook(hook)

    def pattern_at_shape(self, shape, pt):
        p1 = self.p1.pattern_at_shape(shape, pt)
        p2 = self.p2.pattern_at_shape(shape, pt)
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

    def _propagate_hook(self, hook):
        self.p1._propagate_hook(hook)
        self.p2._propagate_hook(hook)
        self.p3._propagate_hook(hook)

    def pattern_at_shape(self, shape, pt):
        self.p1.a = self.p2.pattern_at_shape(shape, pt)
        self.p1.b = self.p3.pattern_at_shape(shape, pt)
        return self.p1.pattern_at_shape(shape, pt)


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

    def _propagate_hook(self, hook):
        self.p1._propagate_hook(hook)

    def pattern_at_shape(self, shape, pt):
        x = pt[0] / self.freq
        y = pt[1] / self.freq
        z = pt[2] / self.freq

        new_x = pt[0] + noise.pnoise3(x, y, z, self.octaves) * self.scale_factor
        z += 1.0
        new_y = pt[1] + noise.pnoise3(x, y, z, self.octaves) * self.scale_factor
        z += 1.0
        new_z = pt[2] + noise.pnoise3(x, y, z, self.octaves) * self.scale_factor

        perturbed_pt = point(new_x, new_y, new_z)
        return self.p1.pattern_at_shape(shape, perturbed_pt)


class Stripe(Pattern):
    def __init__(self, color1=WHITE, color2=BLACK):
        Pattern.__init__(self, color1, color2)

    def pattern_at(self, pattern_point):
        return self._predicate_eval(np.floor(pattern_point[0]) % 2 == 0)


class Gradient(Pattern):
    def __init__(self, color1=WHITE, color2=BLACK):
        Pattern.__init__(self, color1, color2)

    def pattern_at(self, pattern_point):
        distance = self.b - self.a
        fraction = pattern_point[0] - np.floor(pattern_point[0])
        return self.a + distance * fraction


class Ring(Pattern):
    def __init__(self, color1=WHITE, color2=BLACK):
        Pattern.__init__(self, color1, color2)

    def pattern_at(self, pattern_point):
        return self._predicate_eval(
            np.floor(np.sqrt(pattern_point[0] ** 2 + pattern_point[2] ** 2)) % 2 == 0)


class Checker(Pattern):
    def __init__(self, color1=WHITE, color2=BLACK):
        Pattern.__init__(self, color1, color2)

    def pattern_at(self, pattern_point):
        new_point = np.floor(pattern_point.arr)
        s = new_point[0] + new_point[1] + new_point[2]
        return self._predicate_eval(np.isclose((s % 2), 0))


class RadialGradient(Pattern):
    def __init__(self, color1=WHITE, color2=BLACK):
        Pattern.__init__(self, color1, color2)

    def pattern_at(self, pattern_point):
        color_a = self._get_color_1()
        distance = self._get_color_2() - color_a
        mag = np.sqrt(pattern_point[0] ** 2 + pattern_point[2] ** 2)
        fraction = mag - np.floor(mag)
        return color_a + distance * fraction


def stripe_pattern(color1, color2):
    """
    >>> pat = stripe_pattern(WHITE, BLACK)
    >>> pat.a == WHITE
    array([ True,  True,  True])
    >>> pat.b == BLACK
    array([ True,  True,  True])
    >>> pat.pattern_at(point(0,0,0)) == WHITE
    array([ True,  True,  True])
    >>> pat.pattern_at(point(0,1,0)) == WHITE
    array([ True,  True,  True])
    >>> pat.pattern_at(point(0,2,0)) == WHITE
    array([ True,  True,  True])
    >>> pat.pattern_at(point(0,0,1)) == WHITE
    array([ True,  True,  True])
    >>> pat.pattern_at(point(0,0,2)) == WHITE
    array([ True,  True,  True])
    >>> pat.pattern_at(point(0.9,0,0)) == WHITE
    array([ True,  True,  True])
    >>> pat.pattern_at(point(1,0,0)) == BLACK
    array([ True,  True,  True])
    >>> pat.pattern_at(point(-.1,0,0)) == BLACK
    array([ True,  True,  True])
    >>> pat.pattern_at(point(-1,0,0)) == BLACK
    array([ True,  True,  True])
    >>> pat.pattern_at(point(-1.1,0,0)) == WHITE
    array([ True,  True,  True])
    """
    return Stripe(color1, color2)

def gradient_pattern(color1, color2, uv_map=False):
    """
    >>> pat = gradient_pattern(WHITE, BLACK)
    >>> np.isclose(pat.pattern_at(point(0,0,0)), WHITE)
    array([ True,  True,  True])
    >>> np.isclose(pat.pattern_at(point(0.25,0,0)), color(0.75, 0.75, 0.75))
    array([ True,  True,  True])
    >>> np.isclose(pat.pattern_at(point(0.5,0,0)), color(0.5, 0.5, 0.5))
    array([ True,  True,  True])
    >>> np.isclose(pat.pattern_at(point(0.75,0,0)), color(0.25, 0.25, 0.25))
    array([ True,  True,  True])
    """
    return Gradient(color1, color2)

def ring_pattern(color1, color2):
    """
    >>> pat = ring_pattern(WHITE, BLACK)
    >>> np.isclose(pat.pattern_at(point(0,0,0)), WHITE)
    array([ True,  True,  True])
    >>> np.isclose(pat.pattern_at(point(1,0,0)), BLACK)
    array([ True,  True,  True])
    >>> np.isclose(pat.pattern_at(point(0,0,1)), BLACK)
    array([ True,  True,  True])
    >>> np.isclose(pat.pattern_at(point(0.708,0,0.708)), BLACK)
    array([ True,  True,  True])
    """
    return Ring(color1, color2)

def checker_pattern(color1, color2):
    """
    >>> pat = checker_pattern(WHITE, BLACK)
    >>> pat.pattern_at(point(0,0,0)) == WHITE
    array([ True,  True,  True])
    >>> pat = checker_pattern(WHITE, BLACK)
    >>> pat.pattern_at(point(0.99,0,0)) == WHITE
    array([ True,  True,  True])
    >>> pat = checker_pattern(WHITE, BLACK)
    >>> pat.pattern_at(point(1.01,0,0)) == BLACK
    array([ True,  True,  True])

    >>> pat = checker_pattern(WHITE, BLACK)
    >>> np.isclose(pat.pattern_at(point(0,0.99,0)), WHITE)
    array([ True,  True,  True])
    >>> pat = checker_pattern(WHITE, BLACK)
    >>> np.isclose(pat.pattern_at(point(0,1.01,0)), BLACK)
    array([ True,  True,  True])

    >>> pat = checker_pattern(WHITE, BLACK)
    >>> np.isclose(pat.pattern_at(point(0,0,0.99)), WHITE)
    array([ True,  True,  True])
    >>> pat = checker_pattern(WHITE, BLACK)
    >>> np.isclose(pat.pattern_at(point(0,0,1.01)), BLACK)
    array([ True,  True,  True])
    """
    return Checker(color1, color2)

def radial_gradient_pattern(color1, color2):
    return RadialGradient(color1, color2)

def blended_pattern(pattern1=stripe_pattern(WHITE, BLACK), pattern2=None):
    return BlendedPattern(pattern1, pattern2)

def nested_pattern(pattern1=stripe_pattern(WHITE, BLACK), pattern2=None, pattern3=None):
    return NestedPattern(pattern1, pattern2, pattern3)

def perturbed_pattern(pattern, frequency=1.0, scale_factor=0.5, octaves=1):
    return PerturbedPattern(pattern, frequency, scale_factor, octaves)

def uv_map_pattern(pattern):
    return SphereUVMapPattern(pattern)

def test_pattern():
    return Pattern()