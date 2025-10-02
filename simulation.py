from particle import Particle
import math
import random


class Simulation:
    def __init__(self, width, height, num_particles=100):
        self.width = width
        self.height = height
        self.num_particles = num_particles
        self.particles = [
            Particle(random.randint(1, width), height / 2, math.pi)
            for _ in range(num_particles)
        ]

    def update(self, dt):
        for p in self.particles:
            p.update(self.width, self.height, dt)

    def draw(self, surface):
        for p in self.particles:
            p.draw(surface)

    def reset(self):
        self.particles = [
            Particle(random.randint(1, self.width), self.height / 2, math.pi)
            for _ in range(self.num_particles)
        ]
