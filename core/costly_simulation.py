import copy
import random
import numpy as np
from core.zombie import Zombie


class CostlySimulation:
    def __init__(self, simulation_depth, simulation_state, possible_actions, agent_type, board_height, board_width, max_angle, max_velocity, max_hit_points, dt, light_size):
        self.simulation_depth = simulation_depth
        self.simulation_state = simulation_state
        self.possible_actions = possible_actions
        self.agent_type = agent_type
        self.board_height = board_height
        self.board_width = board_width
        self.max_angle = max_angle
        self.max_velocity = max_velocity
        self.max_hit_points = max_hit_points
        self.dt = dt
        self.light_size = light_size

    def costly_simulation(self, height, width):
        simulation_reward = 0
        simulation_state = self.simulation_state
        for __ in range(self.simulation_depth):
            # select and execute next action (taken from simulation node)
            action = self.select_simulation_action()
            one_step_reward, simulation_state = self.simulate_step(simulation_state, action, height, width)
            # aggregate reward
            simulation_reward += one_step_reward
        return simulation_reward

    def select_simulation_action(self):
        # Randomly selects a child node.
        i = random.sample(self.possible_actions, 1)[0]
        return i

    def simulate_step(self, simulation_state, action, height, width):
        alive_zombies = list(copy.deepcopy(simulation_state))  # make a copy of all zombies - we do not want to make any act in real world

        # set action and light agents actions
        if self.agent_type == 'zombie':
            zombie_action = action
            # random sample len(actions) times from light-agent actions-space
            light_action = np.random.randint(0, height * width)
        else:
            light_action = action
            # sample n times from zombie-agent actions-space
            zombie_action = np.random.randint(0, height)

        # simulate reward
        new_zombie = self.create_zombie(zombie_action)
        alive_zombies.append(new_zombie)
        reward, next_alive_zombies = self.calc_reward_and_move_zombies(alive_zombies, light_action)

        return reward, next_alive_zombies

    def calc_reward_and_move_zombies(self, alive_zombies, light_action):
        """
        moving all zombies while aggregating and outputting current reward
        :return all alive zombies (haven't step out of the grid)
        """
        # temp list for later be equal to self.alive_zombies list, it's here just for the for loop (NECESSARY!)
        temp_alive_zombies = list(copy.deepcopy(alive_zombies))
        indices_to_remove = []
        reward = 0
        for index, z in enumerate(temp_alive_zombies):
            z.move(light_action)
            if z.x >= self.board_width:
                if self.keep_alive(z.hit_points):  # decide whether to keep the zombie alive, if so, give the zombie master reward
                    reward += 1
                indices_to_remove.append(index)  # deleting a zombie that reached the border
        for index in indices_to_remove:
            temp_alive_zombies.pop(index)
        return reward, temp_alive_zombies

    def keep_alive(self, h):
        if h >= self.max_hit_points:  # if the zombie sustained a lot of damaged
            return False
        else:  # else decide by the sine function -> if the result is greater than 0.5 -> keep alive, else -> kill it (no reward for the zombie master)
            """
            the idea is: if the hit points is close to 3 then the result is close to 1 ->
             -> there is small chance for keeping him alive and therefore rewarding the zombie with positive reward
             For example, if zombie hit points is 3 - > the result is 1 -> always return False (the random will never be greater than 1)
            in the past sin(h * pi / 2 * self.max_hit_points) < random.random()
            """
            return np.power(h / self.max_hit_points, 1 / 3) < random.random()

    def create_zombie(self, position):
        if self.max_angle == 0:
            angle = self.max_angle
        else:
            angle = random.uniform(-self.max_angle, self.max_angle)
        return Zombie(angle, self.max_velocity, position, self.board_width, self.board_height, self.dt, self.light_size)
