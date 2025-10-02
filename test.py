import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
import math

# --------------------------
# Configuración inicial
# --------------------------
WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Partículas 3D con cámara libre")

clock = pygame.time.Clock()

# Configuración OpenGL
glEnable(GL_DEPTH_TEST)
glPointSize(5)  # hace visibles los puntos
gluPerspective(60, WIDTH / HEIGHT, 0.1, 100.0)
glClearColor(0.0, 0.0, 0.0, 1.0)  # negro

# --------------------------
# Partículas
# --------------------------
NUM_PARTICLES = 200
particles = []

for _ in range(NUM_PARTICLES):
    x = random.uniform(-5, 5)
    y = random.uniform(-5, 5)
    z = random.uniform(-5, 5)
    vx = random.uniform(-1, 1)
    vy = random.uniform(-1, 1)
    vz = random.uniform(-1, 1)
    particles.append([x, y, z, vx, vy, vz])

GRAVITY = -9.8

# --------------------------
# Cámara
# --------------------------
cam_pos = [0.0, 0.0, 10.0]
cam_rot = [0.0, 0.0]  # pitch, yaw
speed = 5.0
mouse_sensitivity = 0.2

pygame.event.set_grab(True)
pygame.mouse.set_visible(False)

# --------------------------
# Loop principal
# --------------------------
running = True
while running:
    dt = clock.tick(60) / 1000.0  # Delta time en segundos

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # --------------------------
    # Teclado
    # --------------------------
    keys = pygame.key.get_pressed()
    # Vector de dirección
    yaw_rad = math.radians(cam_rot[1])
    forward = [math.sin(yaw_rad), 0, -math.cos(yaw_rad)]
    right = [-forward[2], 0, forward[0]]

    if keys[pygame.K_w]:
        cam_pos[0] += forward[0] * speed * dt
        cam_pos[2] += forward[2] * speed * dt
    if keys[pygame.K_s]:
        cam_pos[0] -= forward[0] * speed * dt
        cam_pos[2] -= forward[2] * speed * dt
    if keys[pygame.K_a]:
        cam_pos[0] += right[0] * speed * dt
        cam_pos[2] += right[2] * speed * dt
    if keys[pygame.K_d]:
        cam_pos[0] -= right[0] * speed * dt
        cam_pos[2] -= right[2] * speed * dt
    if keys[pygame.K_q]:
        cam_pos[1] += speed * dt
    if keys[pygame.K_e]:
        cam_pos[1] -= speed * dt

    # --------------------------
    # Mouse
    # --------------------------
    mx, my = pygame.mouse.get_rel()
    cam_rot[1] += mx * mouse_sensitivity  # yaw
    cam_rot[0] += my * mouse_sensitivity  # pitch
    cam_rot[0] = max(-89, min(89, cam_rot[0]))

    # --------------------------
    # Actualizar partículas
    # --------------------------
    for p in particles:
        # Gravedad
        p[4] += GRAVITY * dt
        # Posición
        p[0] += p[3] * dt
        p[1] += p[4] * dt
        p[2] += p[5] * dt
        # Rebote en piso
        if p[1] < -5:
            p[1] = -5
            p[4] *= -0.6  # rebote con pérdida

    # --------------------------
    # Dibujar
    # --------------------------
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # glLoadIdentity()
    # glRotatef(cam_rot[0], 1, 0, 0)  # pitch
    # glRotatef(cam_rot[1], 0, 1, 0)  # yaw
    # glTranslatef(-cam_pos[0], -cam_pos[1], -cam_pos[2])

    glBegin(GL_POINTS)
    glColor3f(1, 1, 1)
    for p in particles:
        glVertex3f(p[0], p[1], p[2])
    glEnd()

    pygame.display.flip()

pygame.quit()
