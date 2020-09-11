import pygame
import numpy as np
from PIL import Image
from environment import gameGrid
import math
import random
from entities import zombie

DISPLAY_WIDTH = 1600
DISPLAY_HEIGHT = 800

# MAX_ANGLE = np.pi / 10  # (float) radians
MAX_ANGLE = 0  # np.arctan(self.grid.get_height()/(2*self.grid.get_width())) in case of 800 and 1600, arctan(1/4) = 0.24497866312686414
MAX_VELOCITY = 1  # (float)
DT = 1  # (int)
MAX_HIT_POINTS = 1


def calculate_start_positions(grid):
    zombie_home_length = int(grid.get_height() - 2 * grid.get_width() * math.tan(MAX_ANGLE))
    zombie_home_start_pos = int(grid.get_height() - zombie_home_length - grid.get_width() * math.tan(MAX_ANGLE))  # m-n-b
    return np.multiply(list(range(zombie_home_start_pos, zombie_home_start_pos + zombie_home_length)), grid.get_width())


class Env:

    def __init__(self, grid: gameGrid, steps_per_episodes, light_size):
        pygame.init()
        pygame.display.set_caption('pickleking')
        self.steps_per_episodes = steps_per_episodes
        self.max_hit_points = MAX_HIT_POINTS
        self.display_width = DISPLAY_WIDTH
        self.display_height = DISPLAY_HEIGHT
        self.light_size = light_size
        self.game_display = pygame.display.set_mode((self.display_width, self.display_height))
        self.clock = pygame.time.Clock()
        self.grid = grid
        self.start_positions = calculate_start_positions(self.grid)
        self.zombie_image, self.light_image, self.grid_image = self.set_up()
        self.current_time = 0
        self.zombie_num = 0
        self.alive_zombies = []  # list of the currently alive zombies
        self.all_zombies = []  # list of all zombies (from all time)
        self.max_angle = MAX_ANGLE
        self.max_velocity = MAX_VELOCITY
        self.dt = DT

    def reset(self):
        self.current_time = 0
        self.zombie_num = 0
        self.alive_zombies = []  # list of the currently alive zombies
        self.all_zombies = []  # list of all zombies (from all time)

    def action_space(self):
        light_action_space = self.grid.get_height() * self.grid.get_width()
        zombie_action_space = len(self.start_positions)
        return light_action_space, zombie_action_space

    def step(self, zombie_action, light_action):
        """
        This method steps the game forward one step and
        shoots a bubble at the given angle.
        Parameters
        ----------
        zombie_action : int
            The action is an angle between 0 and 180 degrees, that
            decides the direction of the bubble.
        light_action

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing the
                state of the environment.
            reward (float) :
                amount of reward achieved by the previous action.
            episode_over (bool) :
                whether it's time to reset the environment again.
        """
        self.current_time += 1
        # add new zombie
        self.add_zombie(zombie_action)
        # update display
        self.update(light_action)
        # damaged_zombies = 0  # for debugging
        reward = 0
        temp_alive_zombies = list(
            np.copy(self.alive_zombies))  # temp list for later be equal to self.alive_zombies list, it's here just for the for loop (NECESSARY!)
        for z in self.alive_zombies:
            temp_hit_points = z.hit_points
            z.move(light_action)
            if z.x >= self.grid.get_width():
                if self.keep_alive(z.hit_points):  # decide whether to keep the zombie alive, if so, give the zombie master reward
                    reward += 1
                # deleting a zombie that reached the border
                temp_alive_zombies.remove(z)
            # elif z.hit_points > temp_hit_points: damaged_zombies += 1
        self.alive_zombies = temp_alive_zombies
        # print(damaged_zombies)
        return self.get_pygame_window(), reward, self.current_time > self.steps_per_episodes  # TODO - maybe pick another terminal condition of the game and assign it to done (as True/False)

    def keep_alive(self, h):
        if h >= self.max_hit_points:  # if the zombie sustained a lot of damaged
            return False
        else:  # else decide by the sine function -> if the result is greater than 0.5 -> keep alive, else -> kill it (no reward for the zombie master)
            """
            the idea is: if the hit points is close to 3 then the result is close to 1 ->
             -> there is small chance for keeping him alive and therefor rewarding the zombie with positive reward
             For example, if zombie hit points is 3 - > the result is 1 -> always return False (the random will never be greater than 1)
            in the past sin(h * pi / 2 * self.max_hit_points) < random.random()
            """
            a = 1
            if h != 1:
                a = np.power(self.max_hit_points, -1 / 5)
            return a * np.power(h, 1 / 5) < random.random()

    def get_pygame_window(self):
        return pygame.surfarray.array3d(pygame.display.get_surface())

    def add_zombie(self, position):
        """
        function for initiate one zombie and
            generate angle, velocity and position of new zombie from the uniform distribution
        :return:
        """

        self.zombie_num += 1
        if self.max_angle == 0:
            angle = self.max_angle
        else:
            angle = random.uniform(-self.max_angle, self.max_angle)

        new_zombie = zombie.Zombie(self.zombie_num, angle, self.max_velocity, position, env=self)

        self.alive_zombies.append(new_zombie)
        self.all_zombies.append(new_zombie)

    def set_up(self):
        # get images
        zombie_image = Image.open('../gameUtils/zombie.png')
        light_image = Image.open('../gameUtils/light.png')
        # resize (light_image is doubled for 2x2 cells)
        zombie_image = zombie_image.resize((int(DISPLAY_WIDTH / self.grid.get_width()), int(DISPLAY_HEIGHT / self.grid.get_height())), 0)
        light_image = light_image.resize(
            (int(DISPLAY_WIDTH / self.grid.get_width()) * self.light_size, int(DISPLAY_HEIGHT / self.grid.get_height()) * self.light_size), 0)
        # save
        zombie_image.save('../gameUtils/zombie_image.png')
        light_image.save('../gameUtils/light_image.png')
        # draw and save the grid
        self.draw_grid()
        # return the images in the pygame format
        return pygame.image.load('../gameUtils/zombie_image.PNG'), pygame.image.load('../gameUtils/light_image.PNG'), pygame.image.load(
            '../gameUtils/grid.jpeg')

    def update(self, light_action):
        event = pygame.event.get()
        self.game_display.blit(self.grid_image, (0, 0))
        x_adjustment = int(self.display_width / self.grid.get_width())
        y_adjustment = int(self.display_height / self.grid.get_height())
        self.game_display.blit(self.light_image, (int(np.mod(light_action, self.grid.get_width()) * x_adjustment),
                                                  int(light_action / self.grid.get_width()) * y_adjustment))
        for z in self.alive_zombies:
            self.game_display.blit(self.zombie_image,
                                   (z.x * x_adjustment, z.y * y_adjustment))
        pygame.display.update()  # better than pygame.display.flip because it can update by param, and not the whole window
        self.clock.tick(30)  # the number of frames per second

    def draw_grid(self):
        x_size = self.display_width / self.grid.get_width()  # x size of the grid block
        y_size = self.display_height / self.grid.get_height()  # y size of the grid block
        for x in range(self.display_width):
            for y in range(self.display_height):
                rect = pygame.Rect(x * x_size, y * y_size,
                                   x_size, y_size)
                pygame.draw.rect(self.game_display, (255, 255, 255), rect, 1)
        # draw the start line
        y_adjustment = int(self.display_height / self.grid.get_height())
        pygame.draw.rect(self.game_display, (0, 200, 50), [0, int((min(self.start_positions))) / self.grid.get_width() * y_adjustment, 10,
                                                           int((max(self.start_positions) + np.diff(self.start_positions)[0] - min(
                                                               self.start_positions))) / self.grid.get_width() * y_adjustment])
        pygame.image.save(self.game_display, '../gameUtils/grid.jpeg')

    def end_game(self):
        pygame.quit()
        quit()
