import numpy as np


class Particle:
    """Estructura de una partícula"""

    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        color: np.ndarray,
        life: float,
        size: float,
        gravity: float,
    ):
        self.position = position
        self.velocity = velocity
        self.color = color
        self.life = life
        self.size = size
        self.gravity = np.array([0.0, gravity, 0.0], dtype=np.float32)

    def update(self, delta_time):
        # Apply gravity
        self.velocity += self.gravity * delta_time

        # Update position
        self.position += self.velocity * delta_time

        # Update life and color
        self.life -= delta_time * 0.5
        self.color[3] = self.life
