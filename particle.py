import pygame
import random
import math

G = 200  # px/s^2


class Particle:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.color = (
            random.randint(150, 255),
            random.randint(150, 255),
            random.randint(150, 255),
        )
        self.mass = random.randint(1, 5)
        self.radius = self.mass * 2

    def update(self, width, height, dt):
        # Movimiento
        # Gravity
        self.vy += G * dt

        # actualizar posición
        self.x += self.vx * dt
        self.y += self.vy * dt

        # colisión con el piso
        if self.y + self.radius > height:
            self.y = height - self.radius
            self.vy *= -0.8  # rebote con pérdida de energía

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
