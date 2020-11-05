from numbers import Number
import numpy as np

EPS = .001


class Vector():
    def __init__(self, x, y, z):
        self.arr = np.array([x, y, z], dtype=np.float64)

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

        if isinstance(other, Number):
            return Vector(self.x + other, self.y + other, self.z + other)

    def __neg__(self):
        return Vector(*(-self.arr))

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, scalar: float):
        return Vector(*(self.arr * scalar))

    def __matmul__(self, other: "Vector") -> float:
        return self.arr @ other.arr

    def __rmul__(self, scalar: float):
        return self * scalar

    def __truediv__(self, scalar: float):
        return self * (1 / scalar)

    def __iter__(self):
        yield from self.arr

    def __str__(self):
        return " ".join(str(x_i) for x_i in self)

    def __repr__(self):
        return str(self)

    def __getitem__(self, index: int):
        return self.arr[index]

    def __setitem__(self, index: int, value: float):
        self.arr[index] = value

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    def near_zero(self) -> bool:
        return all(abs(coord) < EPS for coord in self)

    def norm(self) -> float:
        return (self @ self)**(.5)

    def unit(self):
        return self / self.norm()

    def lerp(self, other, weight_for_self: float):
        return weight_for_self * self + (1 - weight_for_self) * other

    def to_color(self):
        return Color(*self.arr)

    def cross(self, other: "Vector") -> "Vector":
        return Vector(*np.cross(self.arr, other.arr))


class Color(Vector):
    @property
    def r(self):
        return self[0]

    @property
    def g(self):
        return self[1]

    @property
    def b(self):
        return self[2]

    def center(self):
        return ((self.unit() + 1) / 2).to_color()

    def attenuate(self, other: "Color") -> "Color":
        return Color(*(self.arr * other.arr))

    def __str__(self):
        return " ".join(str(int(255.999 * c)) for c in self)


class Ray():
    def __init__(self, base: Vector, direction: Vector) -> None:
        self.base = base
        self.direction = direction

    def __call__(self, distance_along_direction):
        return self.base + (distance_along_direction * self.direction)


def test():
    ...

if __name__ == '__main__':
    test()


