from abc import ABC, abstractmethod
from numpy import sqrt
from vector import Vector, Ray


INF = float('inf')
EPS = 0


class HitRecord():
    def __init__(self, resultant=None, t=INF, normal=None, material=None) -> None:
        self.t = t
        self.resultant = resultant
        self.resultant_outward = False
        self.material = material


    def orient(self, incoming_ray: Ray, outward_normal: Vector) -> None:
        if not self:
            return

        self.resultant_outward = (incoming_ray.direction @ outward_normal) < EPS
        if not self.resultant_outward:
            self.resultant.direction = -self.resultant.direction

    def __bool__(self):
        return self.resultant is not None


class Hittable(ABC):
    def __init__(self, material: "Material"):
        self.material = material
    
    @abstractmethod
    def hit(self, ray: Ray, t_min=0, t_max=INF) -> HitRecord:
        raise NotImplementedError()

class HittableList(Hittable):
    def __init__(self, hittable_items=None) -> None:
        self.items = hittable_items or []

    def add(self, item: Hittable) -> None:
        self.items.append(item)

    def clear(self):
        self.items = []

    def hit(self, ray: Ray, t_min=0, t_max=INF) -> HitRecord:
        if not self.items:
            return HitRecord()

        return min([item.hit(ray, t_min, t_max) for item in self.items],
                   key=lambda x: x.t)


class Sphere(Hittable):
    def __init__(self, center: Vector, radius: float, material: "Material") -> None:
        super().__init__(material)
        self.center = center
        self.radius = radius

    def hit(self, ray: Ray, t_min=0, t_max=INF) -> HitRecord:
        base_to_center = ray.base - self.center

        a = ray.direction @ ray.direction
        b = ray.direction @ base_to_center
        c = base_to_center @ base_to_center - self.radius**2

        discriminant = b**2 - a * c

        # Not hit
        if discriminant < 0:
            return HitRecord()

        sqrt_discriminant = sqrt(discriminant)
        roots = [(-b - sqrt_discriminant) / a,
                 (-b + sqrt_discriminant) / a]

        for sphere_hit_coordinate in roots:
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
