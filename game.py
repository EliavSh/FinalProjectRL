import pygame
import numpy as np
from PIL import Image

DISPLAY_WIDTH = 1600
DISPLAY_HEIGHT = 800


class Game:

    def __init__(self, env):
        pygame.init()
        pygame.display.set_caption('pickleking')

        self.env = env
        self.display_width = DISPLAY_WIDTH
        self.display_height = DISPLAY_HEIGHT
        self.game_display = pygame.display.set_mode((self.display_width, self.display_height))
        self.clock = pygame.time.Clock()
        self.zombie_image, self.light_image, self.grid_image = self.set_up()

    def set_up(self):
        # get images
        zombie_image = Image.open('gameUtils/pickle_rick_tiny.PNG')
        light_image = Image.open('gameUtils/giant-head.PNG')
        # resize (light_image is doubled for 2x2 cells)
        zombie_image = zombie_image.resize((int(DISPLAY_WIDTH / self.env.grid.get_width()), int(DISPLAY_HEIGHT / self.env.grid.get_height())), 0)
        light_image = light_image.resize((int(DISPLAY_WIDTH / self.env.grid.get_width()) * 3, int(DISPLAY_HEIGHT / self.env.grid.get_height()) * 3), 0)
        # save
        zombie_image.save('gameUtils/zombie_image.png')
        light_image.save('gameUtils/light_image.png')
        # draw and save the grid
        self.draw_grid()
        # return the images in the pygame format
        return pygame.image.load('gameUtils/zombie_image.PNG'), pygame.image.load('gameUtils/light_image.PNG'), pygame.image.load('gameUtils/grid.jpeg')

    def update(self, alive_zombies, light_pos):
        event = pygame.event.get()
        self.game_display.blit(self.grid_image, (0, 0))
        x_adjustment = int(self.display_width / self.env.grid.get_width())
        y_adjustment = int(self.display_height / self.env.grid.get_height())
        self.game_display.blit(self.light_image, (int(np.mod(light_pos, self.env.grid.get_width()) * x_adjustment),
                                                  int(light_pos / self.env.grid.get_width()) * y_adjustment))
        for z in alive_zombies:
            self.game_display.blit(self.zombie_image,
                                   (z.x * x_adjustment, z.y * y_adjustment))
        pygame.display.update()  # better than pygame.display.flip because it can update by param, and not the whole window
        self.clock.tick(30)  # the number of frames per second

    def draw_grid(self):
        x_size = self.display_width / self.env.grid.get_width()  # x size of the grid block
        y_size = self.display_height / self.env.grid.get_height()  # y size of the grid block
        for x in range(self.display_width):
            for y in range(self.display_height):
                rect = pygame.Rect(x * x_size, y * y_size,
                                   x_size, y_size)
                pygame.draw.rect(self.game_display, (255, 255, 255), rect, 1)
        # draw the start line
        y_adjustment = int(self.display_height / self.env.grid.get_height())
        pygame.draw.rect(self.game_display, (0, 200, 50), [0, int((min(self.env.start_positions))) / self.env.grid.get_width() * y_adjustment, 10,
                                                           int((max(self.env.start_positions) + np.diff(self.env.start_positions)[0] - min(
                                                               self.env.start_positions))) / self.env.grid.get_width() * y_adjustment])
        pygame.image.save(self.game_display, 'gameUtils/grid.jpeg')

    def end_game(self):
        pygame.quit()
        quit()


"""
        while not crashed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    crashed = True
        
        
        pygame.image.load('gameUtils/pickle_rick_tiny.PNG')
"""
