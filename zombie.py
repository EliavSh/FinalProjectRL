import numpy as np
import env


class Zombie:
    def __init__(self, id, angle, velocity, state, env: env):
        """
        :param id: int
        :param angle: float, radians
        :param velocity: float, unit/sec
        :param y: float
        :param env: env - when creating a zombie, we must specify in which env he is born
        """
        self.id = id
        self.angle = angle
        self.velocity = velocity
        self.strength = 1  # 1 for alive, 0 for dead
        self.env = env
        # x,y are the real coordinates of the zombie
        self.x = 0  # every zombie starts at the left side
        self.v_x = self.velocity * np.cos(self.angle)
        self.y = state / self.env.grid.get_width()  # every zombie starts in an arbitrary positions by some distribution
        self.v_y = self.velocity * np.sin(self.angle)
        self.current_state = state
        self.history = [self.current_state]  # tuples of (timestamp, pos) TODO - make that a tuple

    def step(self):
        """
        update current pos of zombie by its' angle and velocity
        """
        self.x += self.v_x * self.env.dt
        self.y += self.v_y * self.env.dt
        self.current_state = self.x + self.y
        self.history.append(self.current_state)
