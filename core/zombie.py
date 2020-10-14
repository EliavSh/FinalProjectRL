import numpy as np
from environment import game
from runnable_scripts.Utils import get_config

HEAL_EPSILON = 0.01

BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
LIGHT_SIZE = int(get_config("MainInfo")['light_size'])
DT = int(get_config("MainInfo")['dt'])


class Zombie:
    # static field
    ZOMBIE_NUM = 1

    def __init__(self, angle, velocity, state):
        """
        :param id: int
        :param angle: float, radians
        :param velocity: float, unit/sec
        :param y: float
        :param env: env_manager - when creating a zombie, we must specify in which env_manager he is born
        """
        self.id = Zombie.set_id()
        self.angle = angle
        self.velocity = velocity
        self.hit_points = 0  # 1 for alive, 0 for dead
        # x,y are the real coordinates of the zombie
        self.x = 0  # every zombie starts at the left side
        self.v_x = self.velocity * np.cos(self.angle)
        self.y = state / BOARD_WIDTH  # every zombie starts in an arbitrary positions by some distribution
        self.v_y = self.velocity * np.sin(self.angle)
        self.current_state = state
        # self.history = [(self.env.current_time, int(self.current_state[0]))]  # tuples of (timestamp, pos)
        self.heal_epsilon = HEAL_EPSILON
        self.just_born = True

    @staticmethod
    def set_id():
        new_zombie_id = Zombie.ZOMBIE_NUM
        Zombie.ZOMBIE_NUM += 1
        return new_zombie_id

    @staticmethod
    def reset_id():
        Zombie.ZOMBIE_NUM = 0

    def update_hit_points(self, light_action):
        light_x = int(np.mod(light_action, BOARD_WIDTH))
        light_y = int(light_action / BOARD_WIDTH)
        # include only the start (the end is outside the light)
        if (light_x <= self.x < (light_x + LIGHT_SIZE)) & (light_y <= self.y < (light_y + LIGHT_SIZE)):
            # in a case of an hit, increase the zombie's hit points by 1
            self.hit_points += 1
        else:
            # heal the zombie by (1-epsilon)
            self.hit_points *= (1 - self.heal_epsilon)

    def move(self, light_action):
        """
        1. punish/heal the zombie by the position of the light
        2. update current pos of zombie by its' angle and velocity
        3. append history
        """
        if self.just_born:
            # if the zombie just born, don't punish him, wait until the next turn to avoid double punishment # TODO - checking if it is necessary
            # new idea: if the zombie just born, punish him without moving him forward
            self.just_born = False
        else:
            # next step, move forward and punish
            self.x += self.v_x * DT
            self.y += self.v_y * DT
            self.current_state = self.x + self.y * BOARD_HEIGHT
        # hit/heal the zombie
        self.update_hit_points(light_action)

        # append history
        # self.history.append((self.env.current_time, int(self.current_state[0])))
