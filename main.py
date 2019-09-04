from .canvas import *
from .pattern import *
from .renderer import *
from .shape import *
from .vector import *

def mirror():
    side = cube()
    side.transform = translation(np.cos(np.pi/3)/3,0,0) * rotation_z(-np.pi/2) * scaling(0.33, 0.01, 1)
    side.material.diffuse = 0.1
    side.material.specular = 1.0
    side.material.transparency = 0
    side.material.reflective = 1.0
    side.material.refractive_index = 0
    side.material.shininess = 300
    side.material.color = color(0,0.01,0)
    return side

def mirror_group():
    g = group()
    m = mirror()
    g.add_child(m)
    return g

def scope():
    s = group()
    s.transform = translation(0,0,-1) * rotation_z(np.pi/6)
    for n in range(3):
        side = mirror_group()
        side.transform = rotation_z(n * 2 * np.pi/3)
        s.add_child(side)
    return s

cam_w = 250
cam_h = cam_w

ppm_ext = 'ppm'
jpg_ext = 'jpg'




wall_color = color(1, 1, 1)
camera_transform = (point(0, 0, -2.5), point(0, 0, 0), vector(0, 1, 0))



world = default_world()
world.contains = []

world.lights = [point_light(point(0.6,0,0.1), color(1, 1, 1))]


p1 = checker_pattern(WHITE, BLACK)



back_wall = plane()
back_wall.transform = translation(0,0,1) * rotation_x(np.pi/2)
back_wall.material.specular = 0.0
back_wall.material.diffuse = 1.0
back_wall.material.pattern = p1
back_wall.material.transparency = 0.01


world.contains.extend([back_wall, scope()])


cam = camera(cam_w, cam_h, np.pi/3)
cam.transform = view_transform(*camera_transform)


from datetime import datetime, timezone, timedelta

file_name = './kaleidoscope_{}.'
max_iterations = 20
for iteration in range(max_iterations):
    if iteration < 5:
        continue
    file_name = './kaleidoscope_{}.'.format(iteration)
    p1.transform = scaling(0.5, 0.5, 0.5) * rotation_y(float(iteration) * np.pi / float(max_iterations))

    now = datetime.now(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
    posix_timestamp_micros_before = (now - epoch) / timedelta(microseconds=1)

    print('canvas construction start at {}'.format(now))
    # render the result to a canvas.
    ca = render_multi(cam, world, 5)


    now = datetime.now(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
    posix_timestamp_micros_after = (now - epoch) / timedelta(microseconds=1)
    delta = posix_timestamp_micros_after - posix_timestamp_micros_before
    print('canvas constructed in {} seconds.'.format(delta/1000000))


    ppm = construct_ppm(ca)
    #print('ppm constructed')

    with open(file_name + ppm_ext, 'wb') as f:
        f.write(ppm)
    print('{} written'.format(file_name + ppm_ext))

    print()