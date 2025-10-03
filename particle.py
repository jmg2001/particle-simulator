import numpy as np
from utils import *

# CONSTANTS
G = -200  # Gravity

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
        self.gravity = np.array([0.0, gravity, 0.0], dtype=np.float32)

    def update(self, delta_time):
        # Apply gravity
        self.velocity += self.gravity * delta_time

        # Update position
        self.position += self.velocity * delta_time

        # Update life and color
        self.life -= delta_time * 0.5
        self.color[3] = self.life
