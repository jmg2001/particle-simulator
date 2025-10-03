import numpy as np
from utils import *

# CONSTANTS
# Gravedad en unidades de OpenGL (basado en 9.81 m/s²)
gravity_ms2 = 9.81 * 100
world_scale = 1  # 1px = 1cm
gravity = np.array([0.0, -gravity_ms2 * world_scale, 0.0], dtype=np.float32)
G = gravity  # Gravity

# Initial Velocity
INITIAL_VELOCITY = np.array([0, 0, 0], dtype=np.float32)
# Random Color
COLOR = get_random_color()
# Initial Position
INITIAL_POSITION = np.array([0.0, 0.0, 0.0], dtype=np.float32)
# Particle Default Size
SIZE = 20
# Particle Deafault Life
LIFE = 10


class Particle:
    """Estructura de una partícula"""

    def __init__(
        self,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        color: np.ndarray = None,
        life: float = LIFE,
        size: float = SIZE,
        gravity: float = G,
    ):
        self.position = position if position is not None else default_position()
        self.velocity = velocity if velocity is not None else default_velocity()
        self.color = color if color is not None else default_color()
        self.life = life
        self.size = size
        self.gravity = gravity

    def update(self, delta_time):
        # Aplicar damping global para estabilidad
        DAMPING = 0.995

        # Apply gravity
        self.velocity += self.gravity * delta_time

        # Aplicar damping (fricción con el aire)
        self.velocity *= DAMPING

        # Update position
        self.position += self.velocity * delta_time

        # Update life and color
        self.life -= delta_time * 0.5
        self.color[3] = self.life
