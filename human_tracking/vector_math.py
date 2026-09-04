from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

def _add_vectors(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(
        x = first.x + second.x,
        y = first.y + second.y,
        z = first.z + second.z
    )

def _subtract_vectors(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(
        x = first.x - second.x,
        y = first.y - second.y,
        z = first.z - second.z,
    )


def _scale_vector(vector: Vector3, scale: float) -> Vector3:
    return Vector3(
        x = vector.x * scale,
        y = vector.y * scale,
        z = vector.z * scale,
    )


def _vector_norm(vector: Vector3) -> float:
    return math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)