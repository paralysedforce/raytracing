import numpy as np
from vector import Vector, Ray

class Camera():
    def __init__(
            self, image_width: int, image_height: int,
            look_from: Vector, look_at: Vector, v_up=None,
            vfov=90, aperture_width=0, focus_dist=1.0) -> None:

        self.image_width = image_width
        self.image_height = image_height

        self.lens_radius = aperture_width / 2
        self.vfov = vfov
        theta = self.vfov * np.pi / 180
        self.viewport_height = 2.0 * np.tan(theta / 2)
        self.viewport_width = self.viewport_height * self.aspect_ratio

        self.look_from = look_from
        self.look_at = look_at
        self.v_up = v_up or Vector(0, 1, 0)

        self.w = (self.look_from - self.look_at).unit()
        self.u = self.v_up.cross(self.w).unit()
        self.v = self.w.cross(self.u)

        self.horizontal = self.viewport_width * self.u * focus_dist
        self.vertical = self.viewport_height * self.v * focus_dist
        self.lower_left = \
            self.look_from -\
            (self.horizontal / 2) -\
            (self.vertical / 2) -\
            (focus_dist * self.w)

    @property
    def aspect_ratio(self):
        return self.image_width / self.image_height

    def _get_offset(self):
        if self.lens_radius == 0:
            return Vector(0, 0, 0)

        while True:
            x, y = (np.random.random(2) * 2) - 1
            if x**2 + y**2 > 1:
                continue

            return ((x * self.u) + (y * self.v)) * self.lens_radius

    def get_ray(self, horizontal_component, vertical_component):
        offset = self._get_offset()
        base = self.look_from + offset

        direction = self.lower_left +\
            (horizontal_component * self.horizontal) +\
            (vertical_component * self.vertical) -\
            base

        return Ray(base, direction)

