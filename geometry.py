from abc import ABC, abstractmethod
from itertools import count
from typing import List
import numpy as np
from vector import Vector, Ray

from scipy.spatial import KDTree




INF = float('inf')
EPS = .001


class HitRecord():
    def __init__(self, resultant=None, t=INF, normal=None, material=None) -> None:
        self.t = t
        self.resultant = resultant
        self.resultant_outward = False
        self.material = material


    def orient(self, incoming_ray: Ray, outward_normal: Vector) -> None:
        if not self:
            return

        self.resultant_outward = (incoming_ray.direction @ outward_normal) <EPS
        if not self.resultant_outward:
            self.resultant.direction = -self.resultant.direction

    def __bool__(self):
        return self.resultant is not None


class BoundingBox(object):
    def __init__(self, minimum: Vector, maximum: Vector) -> None:
        self.min = minimum
        self.max = maximum

    def hit(self, ray: Ray, t_min=0, t_max=INF) -> bool:
        for i in range(3):
            if ray.direction[i] != 0:
                points = [t_min,
                          (self.min[i] - ray.base[i]) / ray.direction[i],
                          (self.max[i] - ray.base[i]) / ray.direction[i],
                          t_max]
                _, t_min, t_max1, _ = sorted(points)

            if t_min < t_max:
                return False
        return True


class Sphere(object):
    def __init__(self, center: Vector, radius: float, material: "Material") -> None:
        self.material = material
        self.center = center
        self.radius = radius
        self.radsquared = radius**2

    def bounding_box(self):
        return self.center - self.radius

    def hit(self, ray: Ray, t_min=0, t_max=INF) -> HitRecord:
        base_to_center = ray.base - self.center

        a = ray.direction @ ray.direction
        b = ray.direction @ base_to_center
        c = base_to_center @ base_to_center - self.radsquared

        discriminant = b**2 - a * c

        # Not hit
        if discriminant <= 0:
            return HitRecord()

        sqrt_discriminant = np.sqrt(discriminant)

        def roots():
            yield from [
                     (-b - sqrt_discriminant) / a,
                     (-b + sqrt_discriminant) / a]

        for sphere_hit_coordinate in roots():
            in_bounds = t_min < sphere_hit_coordinate < t_max
            if not in_bounds:
                continue

            sphere_boundary = ray(sphere_hit_coordinate)
            outward_normal = (sphere_boundary - self.center) / self.radius
            normal = Ray(sphere_boundary, outward_normal)

            record = HitRecord(
                        resultant=normal,
                        t=sphere_hit_coordinate,
                        material=self.material)

            record.orient(ray, outward_normal)
            return record

        return HitRecord()

class World(object):
    def __init__(self, items=None) -> None:
        self.items = items or []

    def hit(self, ray: Ray, t_min=0, t_max=INF) -> HitRecord:
        if not self.items:
            return HitRecord()

        return min((item.hit(ray, t_min, t_max) for item in self.items),
                   key=lambda collision: collision.t)

