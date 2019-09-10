from pattern import *
from renderer import *
from shapes import *
from vector import *

from datetime import datetime, timezone, timedelta
import numpy as np
import PIL.Image as Image

def doctest():
    import doctest
    doctest.testmod()

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

def cornell_box():
    cam_w = 100
    cam_h = cam_w

    close_up = False

    file_name = './cornellbox16.'
    ppm_ext = 'ppm'
    jpg_ext = 'jpg'

    wall_color = color(1, 0.9, 0.9)
    camera_transform = (point(0, 2.2, -9.5), point(0, 2.2, 1), vector(0, 1, 0))
    if close_up:
        camera_transform = (point(0, 2.0, -3), point(-0.5, 1.0, 1), vector(0, 1, 0))


    world = default_world()
    world.contains = []
    world.lights = [point_light(point(1.5, 4.8, 1.5), color(.1, .1, .1)),
                    point_light(point(0, 4.8, 1.5), color(.1, .1, .1)),
                    point_light(point(-1.5, 4.8, 1.5), color(.1, .1, .1)),
                    point_light(point(1.5, 4.8, 0), color(.1, .1, .1)),
                    point_light(point(0, 4.8, 0), color(.2, .2, .2)),
                    point_light(point(-1.5, 4.8, 0), color(.1, .1, .1)),
                    point_light(point(1.5, 4.8, -1.5), color(.1, .1, .1)),
                    point_light(point(0, 4.8, -1.5), color(.1, .1, .1)),
                    point_light(point(-1.5, 4.8, -1.5), color(.1, .1, .1))]

    world.lights = [point_light(point(0, 4.5, 0), color(1, 1, 1))]
    if close_up:
        world.lights = [point_light(point(0, 2, -1), color(1, 1, 1))]

    p1 = checker_pattern(BLACK, WHITE)
    p1.transform = rotation_x(np.pi/4) #matrix_multiply(translation(np.sqrt(72),0,0), matrix_multiply(rotation_y(np.pi/4), scaling(np.sqrt(72) + 0.5, np.sqrt(72) + 0.5, np.sqrt(72) + 0.5)))

    p2 = stripe_pattern(color(1,0,0), color(1, 0.7, 0.8))
    p2.transform = matrix_multiply(rotation_y(np.pi/4), scaling(0.15,1,1))

    p3 = blended_pattern(p1, p2) # blended_pattern(blended_pattern(p1, p2), ring_pattern(color(0,0,1), color(0,0,0.1)))

    p4 = stripe_pattern(WHITE, color(0.8, 0.8, 0.8))
    p4.transform = matrix_multiply(rotation_y(-np.pi/4), scaling(0.1,1,1))

    p5 = nested_pattern(p1, p2, p4)
    p6 = checker_pattern(WHITE, BLACK)
    p6.transform = shearing(1,0,0,0,0,0) *scaling(0.5,0.5,0.5)#matrix_multiply(rotation_x(-np.pi/4), scaling(0.1,0.1,0.1))
    p7 = perturbed_pattern(p5)

    p8 = perturbed_pattern(p6, scale_factor=0.05, frequency=0.1, octaves=7)#p6#uv_map_pattern(perturbed_pattern(p6, scale_factor=0.2, frequency=0.4, octaves=1))
    #p9 = perturbed_pattern(p2, scale_factor=0.3, frequency=0.4, octaves=1)

    p9 = perturbed_pattern(p1, scale_factor=0.1, frequency=0.4, octaves=1)

    floor = cube()
    floor.material.color = wall_color
    floor.material.specular = 0
    floor.material.diffuse = 0.9
    floor.transform = translation(0,-0.2,0.05) * scaling(3.1, 0.2, 3.2)
    #floor.material.pattern = p9

    ceiling = cube()
    ceiling.material.color = wall_color
    ceiling.material.specular = 0
    ceiling.material.reflective = 0.0
    ceiling.transform = translation(0,5.2,0.01) * scaling(3.1, 0.2, 3.2)

    back_wall = cube()
    back_wall.transform = translation(0,2.5,3.2) * rotation_x(np.pi/2) * scaling(3.1, 0.2, 3)
    back_wall.material.color = wall_color
    back_wall.material.specular = 0
    back_wall.material.pattern = p9


    left_wall = cube()
    left_wall.material.color = color(1,0,0)
    left_wall.material.specular = 0
    left_wall.material.reflective = 0.0
    left_wall.transform = translation(-3.2, 2.5, 0) * rotation_z(-np.pi/2) * scaling(2.8, 0.2, 3.2)


    right_wall = cube()
    right_wall.material.color = color(0,1,0)
    right_wall.material.specular = 0
    right_wall.material.shininess = 300
    right_wall.transform = translation(3.2, 2.5, 0) * rotation_z(np.pi/2) * scaling(2.8, 0.2, 3.2)


    middle = cone()
    middle.closed = True
    middle.minimum = 0.0
    middle.maximum = 1.0
    middle.transform =  translation(-2, 0, 0) * rotation_y(-np.pi/4-0.3) * rotation_x(-np.pi/3) * scaling(1.0, 2.0, 1.0) #* rotation_y(np.pi/6) * rotation_z(np.pi/2)
    middle.material = material()
    middle.material.color = color(1,1,1)
    #middle.material.pattern = p1
    #middle.material.ambient = 0.2

    middle.material.specular = 1.0
    middle.material.transparency = 1.0
    middle.material.shininess = 300.0
    middle.material.reflective = 1.0
    middle.material.refractive_index = 1.5
    middle.material.diffuse = 0.0

    right = sphere()
    right.transform = matrix_multiply(translation(1.5, 0.5, -0.5), scaling(0.5, 0.5, 0.5))
    right.material = material()
    right.material.color = color(0,0,0)
    right.material.diffuse = 0.0
    right.material.specular = 1.0
    right.material.reflective = 1.0
    right.material.shininess = 300


    left = sphere()
    left.transform = translation(1, 3.2, -5.0)
    left.material = material()
    left.material.color = color(0,0,0)
    #left.material.pattern = p8
    left.material.diffuse = 0.0
    left.material.specular = 1.0
    left.material.transparency = 1.0
    left.material.reflectivity = 1.0
    left.material.refractive_index = 1.5
    left.material.shininess = 300


    world.contains.extend([floor, ceiling, left_wall, right_wall, back_wall, middle, right, left])


    cam = camera(cam_w, cam_h, np.pi/3)
    cam.transform = view_transform(*camera_transform)


    now = datetime.now(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
    posix_timestamp_micros_before = (now - epoch) / timedelta(microseconds=1)

    print('canvas construction start at {}'.format(now))
    # render the result to a canvas.
    ca = render_multi(cam, world, 4)


    now = datetime.now(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
    posix_timestamp_micros_after = (now - epoch) / timedelta(microseconds=1)
    delta = posix_timestamp_micros_after - posix_timestamp_micros_before
    print('canvas constructed in {} seconds.'.format(delta/1000000))


    ppm = construct_ppm(ca)
    print('ppm constructed')

    with open(file_name + ppm_ext, 'wb') as f:
        f.write(ppm)
    print('ppm file written')

    im = Image.open(file_name + ppm_ext, 'r')
    im.save(file_name + jpg_ext)


def kaleidoscope():
    cam_w = 768
    cam_h = cam_w

    ppm_ext = 'ppm'

    camera_transform = (point(0, 0, -2.5), point(0, 0, 0), vector(0, 1, 0))



    world = default_world()
    world.contains = []

    world.lights = [point_light(point(0.6,0,0.1), color(1, 1, 1))]

    orange = color(0.8, 0.5, 0.2)
    green = color(0,0,1)
    p1 = gradient_pattern(orange, color(0,1,0))
    p2 = checker_pattern(WHITE, BLACK)
    p3 = radial_gradient_pattern(green, orange)
    p4 = nested_pattern(p2, p1, p3)



    back_wall = plane()
    back_wall.material.specular = 0.0
    back_wall.material.diffuse = 1.0
    back_wall.material.pattern = p4


    world.contains.extend([back_wall, scope()])


    cam = camera(cam_w, cam_h, np.pi/3)
    cam.transform = view_transform(*camera_transform)


    max_iterations = 48
    for iteration in range(max_iterations):
        file_name = './kaleidoscope_{}.'.format(iteration)
        back_wall.transform = translation(0,0,1) * rotation_z(float(iteration) * np.pi / float(max_iterations)) * rotation_x(np.pi/2)

        now = datetime.now(timezone.utc)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
        posix_timestamp_micros_before = (now - epoch) / timedelta(microseconds=1)

        print('canvas construction start at {}'.format(now))
        # render the result to a canvas.
        ca = render_multi(cam, world, 4)


        now = datetime.now(timezone.utc)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
        posix_timestamp_micros_after = (now - epoch) / timedelta(microseconds=1)
        delta = posix_timestamp_micros_after - posix_timestamp_micros_before
        print('canvas constructed in {} seconds.'.format(delta/1000000))


        ppm = construct_ppm(ca)

        with open(file_name + ppm_ext, 'wb') as f:
            f.write(ppm)
        print('{} written'.format(file_name + ppm_ext))

        print()

    # animated gif creation
    file_name = './kaleidoscope_{}.ppm'
    images = [Image.open(file_name.format(n)) for n in range(max_iterations)]

    images[0].save('kaleidoscope.gif',
                   save_all=True,
                   append_images=images[1:],
                   duration=100,
                   loop=0)
    for im in images:
        im.close()

def yaml_test(yaml_file_name, output_file_name):
    from yaml_parser import yaml_file_to_world_objects
    w = default_world()
    objs = yaml_file_to_world_objects(yaml_file_name)
    w.contains = objs['world']
    w.lights = objs['lights']

    now = datetime.now(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
    posix_timestamp_micros_before = (now - epoch) / timedelta(microseconds=1)

    print('canvas construction start at {}'.format(now))
    ca = render_multi(objs['camera'], w, 4)

    now = datetime.now(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
    posix_timestamp_micros_after = (now - epoch) / timedelta(microseconds=1)
    delta = posix_timestamp_micros_after - posix_timestamp_micros_before
    print('canvas constructed in {} seconds.'.format(delta/1000000))

    ppm = construct_ppm(ca)
    with open(output_file_name + 'ppm', 'wb') as f:
        f.write(ppm)

def csg_test(yaml_file_path, output_file_path):
    from yaml_parser import yaml_file_to_world_objects

    w = default_world()
    w.contains = []

    yaml_objs = yaml_file_to_world_objects(yaml_file_path)

    cam = yaml_objs['camera']
    w.lights = yaml_objs['lights']

    gs = yaml_objs['world']

    now = datetime.now(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
    posix_timestamp_micros_before = (now - epoch) / timedelta(microseconds=1)

    print('Bounding volume construction start at {}'.format(now))
    gs[0].divide(1)
    w.contains.extend(yaml_objs['world'])

    now = datetime.now(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
    posix_timestamp_micros_middle = (now - epoch) / timedelta(microseconds=1)
    delta = posix_timestamp_micros_middle - posix_timestamp_micros_before
    print('Bounding volume constructed in {} seconds'.format(delta/1000000))

    print('canvas construction start at {}'.format(now))
    ca = render_multi(cam, w, 4)

    now = datetime.now(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
    posix_timestamp_micros_after = (now - epoch) / timedelta(microseconds=1)
    delta = posix_timestamp_micros_after - posix_timestamp_micros_middle
    print('canvas constructed in {} seconds.'.format(delta/1000000))

    ppm = construct_ppm(ca)
    with open(output_file_path, 'wb') as f:
        f.write(ppm)


if __name__ == '__main__':
    input_yaml_file_path = './bounding_boxes.yml'
    output_file_path = './bounding_boxes.ppm'
    csg_test(input_yaml_file_path, output_file_path)
    #import cProfile
    #cProfile.run('csg_test(input_yaml_file_path, output_file_path)', sort='tottime')

    im = Image.open(output_file_path, 'r')
    im.save(output_file_path[:-3] + 'jpg')
    im.close()
