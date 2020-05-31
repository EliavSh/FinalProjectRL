import numpy as np
import env

HEAL_EPSILON = 0.1


class Zombie:
    def __init__(self, id, angle, velocity, state, env: env):
        """
        :param id: int
        :param angle: float, radians
        :param velocity: float, unit/sec
        :param y: float
        :param env: env_manager - when creating a zombie, we must specify in which env_manager he is born
        """
        self.id = id
        self.angle = angle
        self.velocity = velocity
        self.hit_points = 0  # 1 for alive, 0 for dead
        self.env = env
        # x,y are the real coordinates of the zombie
        self.x = 0  # every zombie starts at the left side
        self.v_x = self.velocity * np.cos(self.angle)
        self.y = state / self.env.grid.get_width()  # every zombie starts in an arbitrary positions by some distribution
        self.v_y = self.velocity * np.sin(self.angle)
        self.current_state = state
        self.history = [(self.env.current_time, self.current_state)]  # tuples of (timestamp, pos) TODO - make that a tuple
        self.heal_epsilon = HEAL_EPSILON
        self.just_born = True

    def step(self, light_action):
        """
        update current pos of zombie by its' angle and velocity
        """
        if self.just_born:  # if the zombie just born, don't take any step
            self.just_born = False
        else:
            self.x += self.v_x * self.env.dt
            self.y += self.v_y * self.env.dt
            self.current_state = self.x + self.y
            light_x = int(np.mod(light_action, self.env.grid.get_width()))
            light_y = int(light_action / self.env.grid.get_width())
            if (light_x <= self.x < (light_x + self.env.light_size)) & (light_y <= self.y < (light_y + self.env.light_size)):  # include only the start (the end is outside the light)
                # in a case of an hit, increase the zombie's hit points by 1
                self.hit_points += 1
            else:
                # heal the zombie by (1-epsilon)
                self.hit_points *= (1 - self.heal_epsilon)
        self.history.append((self.env.current_time, self.current_state))
