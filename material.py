from abc import ABC, abstractmethod
from numpy.random import random
from numpy import sqrt

from geometry import HitRecord
from vector import Ray, Color, Vector

def random_in_sphere() -> Vector:
    while True:
        random_in_cube = random(3) * 2 - 1
        if random_in_cube @ random_in_cube < 1:
            return Vector(*random_in_cube)


def random_in_hemisphere(normal: Vector) -> Vector:
    random_vector = random_in_sphere()
    if random_vector @ normal > 0:
        return random_vector
    else:
        return -random_vector


def random_unit_vector(normal: Vector) -> Vector:
    return random_in_sphere().unit()


class MaterialRecord(object):
    def __init__(self, color=None, outgoing_ray=None, absorbed=False) -> None:
        self.color = color
        self.outgoing_ray = outgoing_ray
        self.absorbed = absorbed

    def __bool__(self):
        return self.outgoing_ray is None


class Material(ABC):
    @abstractmethod
    def scatter(self, ray_in: Ray, collision: HitRecord) -> MaterialRecord:
        raise NotImplementedError()


class Lambertian(Material):
    def __init__(self, albedo: Color) -> None:
        self.albedo = albedo

    def scatter(self, ray_in: Ray, collision: HitRecord) -> MaterialRecord:
        base = collision.resultant.base
        direction = collision.resultant.direction.unit()

        reflected_direction = direction + random_in_hemisphere(direction)
        if reflected_direction.near_zero():
            reflected_direction = direction

        reflected_ray = Ray(base, reflected_direction)
        color = self.albedo

        return MaterialRecord(outgoing_ray=reflected_ray, color=color)


def reflect_across(incoming_direction: Vector, normal: Vector) -> Vector:
    return incoming_direction - 2 * (incoming_direction @ normal) * normal


def refract_across(incoming_direction: Vector, normal: Vector, index_ratio) -> Vector:

    cos_theta = min(-incoming_direction @ normal, 1)

    perp_component = index_ratio * (incoming_direction + cos_theta * normal)
    parallel_component = -normal * sqrt(max(1 - perp_component.norm()**2, 0))
    return perp_component + parallel_component



class Metal(Material):
    def __init__(self, albedo: Color, fuzziness=0) -> None:
        self.albedo = albedo
        self.fuzziness = min(fuzziness, 1)

    def scatter(self, ray_in: Ray, collision: HitRecord) -> MaterialRecord:
        base = collision.resultant.base
        incoming_direction = ray_in.direction.unit()
        normal = collision.resultant.direction.unit()
        fuzz = random_in_sphere() * self.fuzziness

        reflected_direction = reflect_across(incoming_direction, normal) + fuzz
        if reflected_direction.near_zero():
            reflected_direction = direction

        reflected_ray = Ray(base, reflected_direction)
        color = self.albedo
        absorbed = (reflected_direction @ normal) < 0

        return MaterialRecord(outgoing_ray=reflected_ray,
                              color=color,
                              absorbed=absorbed)


class Dielectric(Material):
    def __init__(self, index_of_refraction: float) -> None:
        self.index_of_refraction = index_of_refraction

    def scatter(self, ray_in: Ray, collision: HitRecord) -> MaterialRecord:
        color = Color(1, 1, 1)

        if collision.resultant_outward:
            incoming_index, outgoing_index = 1.0, self.index_of_refraction
        else:
            incoming_index, outgoing_index = self.index_of_refraction, 1.0

        index_ratio = incoming_index / outgoing_index
        direction = ray_in.direction.unit()
        normal = collision.resultant.direction

        cos_theta = -direction @ normal
        sin_theta = sqrt(max(1 - cos_theta**2, 0))

        # Use the Schlick approximation to calculate reflectance
        r_0 = ((1 - index_ratio) / (1 + index_ratio))**2
        reflectance = r_0 + (1 - r_0) * (1 - cos_theta)**5

        snell_solvable = sin_theta * index_ratio <= 1.0
        schlick_within_range = reflectance <= random()
        can_refract = snell_solvable and schlick_within_range

        if can_refract:
            outgoing_direction = refract_across(direction, normal, index_ratio)
        else:
            outgoing_direction = reflect_across(direction, normal)

        outgoing_ray = Ray(collision.resultant.base, outgoing_direction)
        return MaterialRecord(color=color, outgoing_ray=outgoing_ray)


