import math
import numpy as np

HEAL_EPSILON = 0.01


def calculate_start_positions(BOARD_WIDTH, BOARD_HEIGHT, ANGLE):
    zombie_home_length = int(BOARD_HEIGHT - 2 * BOARD_WIDTH * math.tan(math.pi * ANGLE / 180))
    zombie_home_start_pos = int(BOARD_HEIGHT - zombie_home_length - BOARD_WIDTH * math.tan(math.pi * ANGLE / 180))  # m-n-b
    return np.multiply(list(range(zombie_home_start_pos, zombie_home_start_pos + zombie_home_length)), BOARD_WIDTH)


class Zombie:
    # static field
    ZOMBIE_NUM = 1

    def __init__(self, angle, velocity, state, board_width, board_height, dt, light_size):
        """
        :param board_width:
        :param board_height:
        :param dt:
        :param light_size:
        :param angle: float, radians
        :param velocity: float, unit/sec
        """

        self.id = Zombie.set_id()
        self.angle = angle
        self.velocity = velocity
        self.hit_points = 0  # 1 for alive, 0 for dead
        # x,y are the real coordinates of the zombie
        self.x = 0  # every zombie starts at the left side
        self.v_x = self.velocity * np.cos(self.angle)
        # every zombie starts in an arbitrary positions by some distribution
        self.y = calculate_start_positions(board_width, board_height, angle)[state] / board_width
        self.v_y = self.velocity * np.sin(self.angle)
        self.current_state = state
        # self.history = [(self.env.current_time, int(self.current_state[0]))]  # tuples of (timestamp, pos)
        self.heal_epsilon = HEAL_EPSILON
        self.just_born = True
        self.dt = dt
        self.light_size = light_size
        self.board_height = board_height
        self.board_width = board_width

    @staticmethod
    def set_id():
        new_zombie_id = Zombie.ZOMBIE_NUM
        Zombie.ZOMBIE_NUM += 1
        return new_zombie_id

    @staticmethod
    def reset_id():
        Zombie.ZOMBIE_NUM = 1

    def update_hit_points(self, light_action):
        light_x = int(np.mod(light_action, self.board_width))
        light_y = int(light_action / self.board_width)
        # include only the start (the end is outside the light)
        if (light_x <= self.x < (light_x + self.light_size)) & (light_y <= self.y < (light_y + self.light_size)):
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
            self.x += self.v_x * self.dt
            self.y += self.v_y * self.dt
            self.current_state = self.x + self.y * self.board_height
        # hit/heal the zombie
        self.update_hit_points(light_action)

        # append history
        # self.history.append((self.env.current_time, int(self.current_state[0])))
