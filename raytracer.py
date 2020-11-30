import sys
import os
import time
from typing import List
import pickle

import numpy as np
from numpy.random import random
from tqdm import tqdm
from PIL import Image

from vector import Ray, Vector, Color
from geometry import Sphere, World
from camera import Camera
from material import Lambertian, Metal, Dielectric, Material


def debug(*message):
    print(*message, file=sys.stderr)


def get_color(ray: Ray, world: List[Sphere], depth: int) -> Color:
    if depth <= 0:
        return Color(0, 0, 0)

    collision = world.hit(ray, t_min=.001)
    if collision:
        scattering = collision.material.scatter(ray, collision)
        if scattering.absorbed:
            return Color(0, 0, 0)

        return scattering.color.attenuate(
                get_color(scattering.outgoing_ray, world, depth-1))

    direction = ray.direction.unit()
    weight_for_blue = (direction.y + 1) / 2
    white = Color(1.0, 1.0, 1.0)
    blue = Color(0.5, 0.7, 1.0)
    return blue.lerp(white, weight_for_blue).to_color()


def clamp(color: Color, samples_per_pixel: int) -> Color:
    for i in range(3):
        color[i] = min(.999, max(0, np.sqrt(color[i] / samples_per_pixel)))
    return color.to_color()


def get_world() -> World:
    world = []
    glass = Dielectric(1.5)

    # Ground
    ground_mat = Lambertian(Color(0.5, 0.5, 0.5))
    world.append(Sphere(Vector(0, -1000, 0), 1000, ground_mat))

    # Marbles
    for x in range(-11, 11):
        for z in range(-11, 11):
            choose_material = random()
            material: Material

            center = Vector(x + .9 * random(), .2, z + .9 * random())

            if (center - Vector(4, .2, 0)).norm() <= .9:
                continue

            if choose_material < .8:
                # Diffuse
                albedo = Color(*(random(3) * random(3)))
                material = Lambertian(albedo)
            elif choose_material < .95:
                # Metal
                albedo = Color(*(random(3) * .5 + .5))
                fuzz = random() * .4
                material = Metal(albedo, fuzziness=fuzz)
            else:
                # Glass
                material = glass

            world.append(Sphere(center, .2, material))

    # Big Spheres
    world.append(Sphere(Vector(4, 1, 0),  1.0, glass))
    world.append(Sphere(Vector(-4, 1, 0), 1.0, Lambertian(Color(.6, .1, .1))))
    world.append(Sphere(Vector(0, 1, 0),  1.0, Metal(Color(.7, .6, .5))))

    return World(world)

def get_small_world() -> World:
    mat_ground = Lambertian(Color(.8, .8, .3))
    ground = Sphere(Vector(0, -100.5, -1), 100, mat_ground)

    mat_center = Lambertian(Color(.7, .3, .3))
    center = Sphere(Vector(0, 0, -1), .5, mat_center)

    mat_left = Metal(Color(.8, .8, .8))
    left = Sphere(Vector(-1, 0, -1), .5, mat_left)

    mat_right = Dielectric(1.5)
    right = Sphere(Vector(1, 0, -1), .5, mat_right)

    return World([ground, center, left, right])


def get_colors():
    checkpointing = True
    checkpoint_filename = "checkpoints/image_in_progress.npy"
    world_filename = "checkpoints/world.pickle"

    aspect_ratio = 1.6
    image_height = 600
    image_width = int(aspect_ratio * image_height)

    samples_per_pixel = 8
    alias_block_size = np.ceil(np.sqrt(samples_per_pixel))
    max_depth = 15

    look_from, look_at = Vector(13, 2, 3), Vector(0, 0, 0)
    focus_dist = 10
    camera = Camera(image_width, image_height,
                    look_from, look_at,
                    vfov=20, aperture_width=.1, focus_dist=focus_dist)

    if checkpointing and os.path.exists(checkpoint_filename):
        colors = np.load(checkpoint_filename)
    else:
        colors = np.ones((image_width, image_height, 3)) * -1

    if checkpointing:
        if os.path.exists(world_filename):
            with open(world_filename, 'rb') as f:
                world = World(pickle.load(f))
        else:
            world = get_world()
            with open(world_filename, 'wb') as f:
                pickle.dump(world.items, f)
    else:
        world = get_world()

    for j in tqdm(range(image_height)):
        for i in tqdm(range(image_width)):

            if colors[i, j, 0] != -1:
                continue

            color = Vector(0, 0, 0)
            for offset in range(samples_per_pixel):
                dx = (offset % alias_block_size) / alias_block_size
                dy = (offset / alias_block_size) / samples_per_pixel

                horizontal_component = (i + dx) / (image_width - 1)
                vertical_component = (j + dy) / (image_height - 1)
                ray = camera.get_ray(horizontal_component, vertical_component)
                color += get_color(ray, world, max_depth)

            color = clamp(color, samples_per_pixel)
            colors[i, j, :] = color.arr

        if checkpointing:
            np.save(checkpoint_filename, colors)

    if checkpointing:
        os.remove(checkpoint_filename)
        os.remove(world_filename)
    return colors

def colors_to_png(colors):
    png_array = np.floor(colors * 256).astype(np.uint8)
    png_array = np.rot90(png_array, 1, axes=(0, 1))
    image = Image.fromarray(png_array)
    image.save('pil_image.png')


def print_colors(colors):
    image_width, image_height, _ = colors.shape

    print("P3")
    print(image_width, image_height)
    print(255)
    for j in range(image_height-1, -1, -1):
        for i in range(image_width):
            color_arr = colors[i, j]
            print(Color(*color_arr))


def main():
    start = time.time()
    colors_to_png(get_colors())
    print(time.time() - start)


if __name__ == '__main__':
    main()
