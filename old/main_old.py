import pygame
from old.simulation_old import Simulation
from old.ui import UI

pygame.init()

# Window Setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulation")

clock = pygame.time.Clock()

# Instances
sim = Simulation(WIDTH, HEIGHT, num_particles=10)
ui = UI()

# Main loop
running = True
while running:
    dt = clock.tick(60) / 1000.0

    # Eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                sim.reset()

    # Update
    sim.update(dt)

    # Render
    screen.fill((20, 20, 30))
    sim.draw(screen)
    ui.draw_text(screen, f"FPS: {int(clock.get_fps())}", 10, 10)

    pygame.display.flip()


pygame.quit()
