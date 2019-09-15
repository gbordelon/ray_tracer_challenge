import numpy as np
from multiprocessing import sharedctypes
from PIL import Image
from functools import lru_cache

CACHE_SIZE = 65536

def color(r, g, b):
    """
    >>> c = color(-0.5, 0.4, 1.7)
    >>> c[0] == -0.5 and c[1] == 0.4 and c[2] == 1.7
    True

    Add colors:
    >>> c1 = color(0.9, 0.6, 0.75)
    >>> c2 = color(0.7, 0.1, 0.25)
    >>> c1 + c2 == color(1.6, 0.7, 1.0)
    array([ True,  True,  True])

    Subtract colors:
    >>> c1 = color(0.9, 0.6, 0.75)
    >>> c2 = color(0.7, 0.1, 0.25)
    >>> c1 - c2
    array([0.2, 0.5, 0.5])

    Multiply color by a scalar:
    >>> c = color(0.2, 0.3, 0.4)
    >>> c * 2 == color(0.4, 0.6, 0.8)
    array([ True,  True,  True])

    Multiply colors:
    >>> c1 = color(1, 0.2, 0.4)
    >>> c2 = color(0.9, 1, 0.1)
    >>> c1 * c2
    array([0.9 , 0.2 , 0.04])
    """
    return np.array([r,g,b], dtype=np.float64)


class TextureCanvas(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.img = Image.open(file_path, 'r')
        self.pixels = np.array(self.img)
        self.scaling_factor = np.float64(np.max(self.pixels))
        self.width = self.pixels.shape[1]
        self.height = self.pixels.shape[0]

    @lru_cache(maxsize=CACHE_SIZE)
    def pixel_at(self, x, y):
        raw_color = self.pixels[x, y]
        r = np.float64(raw_color[0]) / self.scaling_factor
        g = np.float64(raw_color[1]) / self.scaling_factor
        b = np.float64(raw_color[2]) / self.scaling_factor
        return color(r, g, b)

class Canvas(object):
    def __init__(self, width, height):
        self.shape = (width, height)
        self.np_arr = np.ctypeslib.as_ctypes(np.array([[color(0,0,0) for y in range(height)] for x in range(width)]))
        self.shared_arr = sharedctypes.RawArray(self.np_arr._type_, self.np_arr)

    def __getitem__(self, index):
        return np.ctypeslib.as_array(self.shared_arr)[index]

    def __setitem__(self, index, value):
        np.ctypeslib.as_array(self.shared_arr)[index] = value

"""
class Canvas(object):
    def __init__(self, width, height):
        self.shape = (width, height)
        self.np_arr = np.zeros((width, height, 3))

    def __getitem__(self, index):
        return self.np_arr[index]

    def __setitem__(self, index, value):
        self.np_arr[index] = value
"""

def canvas(w, h):
    """
    >>> ca = canvas(10, 20)
    >>> ca[0, 0] == color(0,0,0)
    array([ True,  True,  True])
    >>> ca[9, 19] == color(0,0,0)
    array([ True,  True,  True])

    """
    c = Canvas(w, h)
    return c

def pixel_at(canvas, x, y):
    return canvas[x, y]

def write_pixels(canvas, block, x_start, y_start, shape):
    arr = np.ctypeslib.as_array(canvas.shared_arr)
    arr[x_start:x_start+shape, y_start:y_start+shape] = block

def write_pixel(canvas, x, y, color):
    """
    Writing pixels to a canvas:
    >>> ca = canvas(10,20)
    >>> red = color(1,0,0)
    >>> write_pixel(ca, 2, 3, red)
    >>> pixel_at(ca, 2, 3) == red
    array([ True,  True,  True])
    """
    if x < 0 or x >= canvas.shape[0]:
        return
    if y < 0 or y >= canvas.shape[1]:
        return
    canvas[x, y] = color

def construct_ppm_header(canvas):
    """
    >>> ca = canvas(5,3)
    >>> ppm = construct_ppm_header(ca)
    >>> lines = ppm.splitlines()
    >>> lines[0]
    'P6'
    >>> lines[1]
    '5 3'
    >>> lines[2]
    '255'
    """
    return """P6
{} {}
255
""".format(canvas.shape[0], canvas.shape[1])

def construct_ppm_body(canvas, default_clamping=False):
    body = bytearray()
    max_component = 1.0

    if not default_clamping:
        if type(canvas) == np.ndarray:
            max_component = np.max(canvas)
        elif type(canvas) == Canvas:
            max_component = np.max(np.ctypeslib.as_array(canvas.shared_arr))
        else:
            print(type(canvas))
            max_component = 1.0

    for y in range(canvas.shape[1]):
        for x in range(canvas.shape[0]):
            for component in canvas[x, y]:
                if component >= max_component:
                    scaled = 255
                elif component <= 0:
                    scaled = 0
                else:
                    scaled = (component * 255 / max_component + 0.5).astype(int)
                body.append(scaled)
    return body

def construct_ppm(canvas, default_clamping=True):
    """
    >>> ca = canvas(5,3)
    >>> c1 = color(1.5,0,0)
    >>> c2 = color(0,0.5,0)
    >>> c3 = color(-0.5,0,1)
    >>> write_pixel(ca, 0, 0, c1)
    >>> write_pixel(ca, 2, 1, c2)
    >>> write_pixel(ca, 4, 2, c3)
    >>> ppm = construct_ppm(ca)
    >>> lines = ppm.splitlines()
    >>> lines[3] == b'\\xff\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x80\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\xff'
    True

    PPM finished with a newline:
    >>> ca = canvas(64, 64)
    >>> ppm = construct_ppm(ca)
    >>> ppm[-1] == 10
    True
    """
    header = construct_ppm_header(canvas)
    body = construct_ppm_body(canvas, default_clamping)
    body.append(10)
    return b"".join([header.encode('utf-8'),body])

def dummy_2():
    pass