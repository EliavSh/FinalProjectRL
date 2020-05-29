import numpy as np
import math
import random
import gameGrid
import zombie

np.random.seed(679)
random.seed(679)

"""
Assumptions - data types:
angle: float in range [-zombie_max_angle, zombie_max_angle] radians
current position of zombies: int in range [0, grid_height * grid_width - 1]

the zombie home is positioned in the middle of the left side of the board.
"""
# MAX_ANGLE = np.pi / 10  # (float) radians
MAX_ANGLE = 0  # np.arctan(self.grid.get_height()/(2*self.grid.get_width())) in case of 800 and 1600, arctan(1/4) = 0.24497866312686414
MAX_VELOCITY = 1  # (float)
DT = 1  # (int)


def calculate_start_positions(grid):
    zombie_home_length = int(grid.get_height() - 2 * grid.get_width() * math.tan(MAX_ANGLE))
    zombie_home_start_pos = int(grid.get_height() - zombie_home_length - grid.get_width() * math.tan(MAX_ANGLE))  # m-n-b
    return np.multiply(list(range(zombie_home_start_pos, zombie_home_start_pos + zombie_home_length)), grid.get_width())


class Env:
    def __init__(self, grid: gameGrid):
        """

        :param grid: (class: Grid)
        """
        self.num_of_zombies = 0  # for setting id to every zombie
        self.alive_zombies = []  # list of the currently alive zombies
        self.all_zombies = []  # list of all zombies (from all time)

        # set grid borders and initialize to zero
        self.grid = grid
        self.start_positions = calculate_start_positions(self.grid)

        self.dt = DT
        self.current_time = 0

        self.zombie_num = 0
        self.max_angle = MAX_ANGLE
        self.max_velocity = MAX_VELOCITY

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

    def step(self):
        """
        here we are moving forward all zombies for one time stamp by their kinematic:
        loop over all zombies and apply the step (zombie) function on them
            if a zombie has died (crossed the border), add to dead list and remove from alive list
        :return:
        """
        for z in self.alive_zombies:
            z.step()
            if z.x >= self.grid.get_width():
                # deleting a zombie that reached the border
                self.alive_zombies.remove(z)
        alive_zombies = [int(self.alive_zombies[i].current_state) for i in range(len(self.alive_zombies))]
        all_zombies = [int(self.all_zombies[i].current_state) for i in range(len(self.all_zombies))]
        print("alive zombies at:", alive_zombies)
        print("all zombies at:", all_zombies)

    def get_reward(self, light_action):
        """
        gets the position of light and returns a positive reward in case of hitting a zombie
        :param light_action: some state in the grid
        :return: 1 for hitting a zombie, 0 otherwise
        """
        boolean = light_action in [i.current_state for i in self.alive_zombies]
        print("zombie hit:", boolean)
        return boolean

    def get_history(self):
        """
        function for retrieving all zombies' histories for the entire simulation
        :return:
        """
