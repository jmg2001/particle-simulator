import pygame


class UI:
    def __init__(self, font_size=20):
        self.font = pygame.font.SysFont("Arial", font_size)

    def draw_text(self, surface, text, x, y, color=(255, 255, 255)):
        img = self.font.render(text, True, color)
        surface.blit(img, (x, y))
