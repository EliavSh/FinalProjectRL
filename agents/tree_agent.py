import copy
import random

from environment.game import Game
from agents.agent import Agent
from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy
from runnable_scripts.Utils import get_config
from core.node import Node
import numpy as np

MAX_HIT_POINTS = int(get_config("MainInfo")['max_hit_points'])
MAX_ANGLE = int(get_config("MainInfo")['max_angle'])
MAX_VELOCITY = int(get_config("MainInfo")['max_velocity'])
BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])


class TreeAgent(Agent):

    def __init__(self, device, agent_type):
        super().__init__(EpsilonGreedyStrategy(), agent_type)
        self.root = Node([])
        self.temporary_root = self.root
        self.leaf_expanded = self.root
        self.current_step = 0
        self.simulation_reward = 0
        self.simulation_num = int(get_config("TreeAgentInfo")['simulation_num'])  # number of simulations in the simulation phase
        self.simulation_depth = int(get_config("TreeAgentInfo")['simulation_depth'])  # number of times to expand a node in single simulation
        self.possible_actions = list(range(BOARD_HEIGHT)) if self.agent_type == 'zombie' else list(range(BOARD_HEIGHT * BOARD_WIDTH))

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1

        # selection phase
        selected_child = self.selection()

        # expansion phase
        action = self.expansion(selected_child)
        self.temporary_root = self.temporary_root.children[action]

        return action, rate, self.current_step

    def learn(self, _, action, __, ___):

        # simulation phase
        self.simulation(action)

        # back-propagation phase, start back-propagating from the child of the leaf we expanded
        TreeAgent.back_propagation(self.leaf_expanded.children[action], self.simulation_reward)

    def selection(self):
        selected_child = self.temporary_root

        # Check if child nodes exist.
        if len(selected_child.children) > 0:
            HasChild = True
        else:
            HasChild = False

        while HasChild:
            selected_child = self.select_child(selected_child)
            if len(selected_child.children) == 0:
                HasChild = False

        return selected_child

    # -----------------------------------------------------------------------#
    # Description:
    #	Given a node, selects the first unvisited child node, or if all
    # 	children are visited, selects the node with greatest UTC value.
    # node	- node from which to select child node from.
    # -----------------------------------------------------------------------#

    def select_child(self, node):
        selected_child = node
        if len(node.children) == 0:
            return node

        for child in node.children:
            if child.visits > 0.0:
                continue
            else:
                return child

        max_weight = 0.0
        for child in node.children:
            # Weight = self.EvalUTC(Child)
            weight = child.sputc
            if weight > max_weight:
                max_weight = weight
                selected_child = child
        return selected_child

    # -----------------------------------------------------------------------#
    # Description: Performs expansion phase of the MCTS.
    # Leaf	- Leaf node to expand.
    # -----------------------------------------------------------------------#

    def expansion(self, leaf):
        self.leaf_expanded = leaf
        # if leaf.visits == 0 and leaf.parent is not None:
        #     return leaf
        # else:
        # Expand.
        if len(leaf.children) == 0:
            children = self.eval_children(leaf)
            for new_child in children:
                leaf.append_child(new_child)

        return self.select_expansion_action(leaf.state, self.possible_actions)

    # -----------------------------------------------------------------------#
    # Description:
    #	Evaluates all the possible children states given a node state
    #	and returns the possible children Nodes.
    # node	- node from which to evaluate children.
    # -----------------------------------------------------------------------#
    def eval_children(self, node):
        # evaluate all possible next states
        next_states = []
        for action in self.possible_actions:
            _, alive_zombies = TreeAgent.simulate_action(node.state, self.agent_type, action)
            next_states.append(alive_zombies)
        children = []
        for state in next_states:
            child_node = Node(state, node)
            children.append(child_node)

        return children

    # -----------------------------------------------------------------------#
    # Description:
    #	Selects a child node randomly.
    # node	- node from which to select a random child.
    # -----------------------------------------------------------------------#
    @staticmethod
    def select_expansion_action(alive_zombies, possible_actions):
        # Randomly selects a child node.
        i = random.sample(possible_actions, 1)[0]
        return i

    # -----------------------------------------------------------------------#
    # Description:
    #	Performs the simulation phase of the MCTS.
    #   for now, the agent takes random actions during the simulation
    # node	- node from which to perform simulation.
    # -----------------------------------------------------------------------#
    def simulation(self, action):
        current_state = self.leaf_expanded.children[action].state

        # Perform simulation.
        total_reward = []
        for _ in range(self.simulation_num):
            simulation_reward = 0
            for _ in range(self.simulation_depth):
                action = TreeAgent.select_expansion_action(current_state, self.possible_actions)
                one_step_reward, _ = TreeAgent.simulate_action(current_state, self.agent_type, action)
                simulation_reward += one_step_reward
            total_reward.append(simulation_reward)

        self.simulation_reward = np.average(total_reward) if self.agent_type == 'zombie' else -1 * np.average(total_reward)

    @staticmethod
    def simulate_action(alive_zombies, agent_type, action):
        """
        Simulating future states by 'actions' of an agent
        :param alive_zombies: all alive zombies at the real world
        :param agent_type: 'zombie' or 'light' agent
        :param action: array containing all the actions to simulate
        :return: total reward of the simulation
        """
        alive_zombies = list(copy.deepcopy(alive_zombies))  # make a copy of all zombies - we do not want to make any act in real world

        # set action and light agents actions
        if agent_type == 'zombie':
            zombie_action = action
            # random sample len(actions) times from light-agent actions-space
            light_action = np.random.randint(0, BOARD_HEIGHT * BOARD_WIDTH)
        else:
            light_action = action
            # sample n times from zombie-agent actions-space
            zombie_action = np.random.randint(0, BOARD_HEIGHT)

        # simulate and aggregate reward
        total_reward = 0
        new_zombie = Game.create_zombie(zombie_action)
        alive_zombies.append(new_zombie)
        reward, alive_zombies = Game.calc_reward_and_move_zombies(alive_zombies, light_action)
        total_reward += reward

        return total_reward, alive_zombies

    @staticmethod
    def back_propagation(node, result):
        # Update node's weight.
        CurrentNode = node
        CurrentNode.wins += result
        CurrentNode.ressq += result ** 2
        CurrentNode.visits += 1
        TreeAgent.EvalUTC(CurrentNode)

        while TreeAgent.HasParent(CurrentNode):
            # Update parent node's weight.
            CurrentNode = CurrentNode.parent
            CurrentNode.wins += result
            CurrentNode.ressq += result ** 2
            CurrentNode.visits += 1
            TreeAgent.EvalUTC(CurrentNode)

    @staticmethod
    def EvalUTC(node):
        # c = np.sqrt(2)
        c = 0.5
        w = node.wins
        n = node.visits
        sumsq = node.ressq
        if node.parent is None:
            t = node.visits
        else:
            t = node.parent.visits

        # avoid log of 0 with: 't or 1'
        UTC = w / n + c * np.sqrt(np.log(t or 1) / n)
        D = 10000.
        Modification = np.sqrt((sumsq - n * (w / n) ** 2 + D) / n)
        # print "Original", UTC
        # print "Mod", Modification
        node.sputc = UTC + Modification

    @staticmethod
    def HasParent(node):
        if node.parent is None:
            return False
        else:
            return True

    def reset(self):
        self.temporary_root = self.root
