import copy
import os
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
C = float(get_config("TreeAgentInfo")['exploration_const'])


class TreeAgent(Agent):

    def __init__(self, device, agent_type):
        super().__init__(EpsilonGreedyStrategy(), agent_type)
        self.possible_actions = list(range(BOARD_HEIGHT)) if self.agent_type == 'zombie' else list(range(BOARD_HEIGHT * BOARD_WIDTH))
        self.root = Node([], self.possible_actions)
        self.temporary_root = self.root  # TODO - change its name to something like: real world state-node
        self.current_step = 0
        self.simulation_reward = 0
        self.simulation_num = int(get_config("TreeAgentInfo")['simulation_num'])  # number of simulations in the simulation phase
        self.simulation_depth = int(get_config("TreeAgentInfo")['simulation_depth'])  # number of times to expand a node in single simulation

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1

        # select next action
        action = self.select_expansion_action(self.temporary_root, self.possible_actions)
        self.eval_children(self.temporary_root, [action])
        self.temporary_root = self.temporary_root.children[action]

        # selection phase
        selected_child = self.selection()

        # expansion phase, here we selecting the action from which we will simulate the selected_child play-out
        # action = self.expansion_all_children(selected_child)
        expansion_action = self.expansion_one_child(selected_child)

        # simulation phase
        self.simulation(expansion_action, selected_child)

        return action, rate, self.current_step

    def learn(self, _, action, __, reward):
        # back-propagation phase, start back-propagating from the current real world node
        TreeAgent.back_propagation(self.temporary_root, reward)

    def selection(self):
        selected_child = self.temporary_root

        # Check if child nodes exist.
        if selected_child.num_children > 0:
            HasChild = True
        else:
            HasChild = False

        while HasChild:
            selected_child = TreeAgent.select_child(selected_child)
            if selected_child.num_children == 0 or selected_child.is_terminal or selected_child.simulated_node:
                HasChild = False

        return selected_child

    # -----------------------------------------------------------------------#
    # Description:
    #	Given a node, selects the first unvisited child node, or if all
    # 	children are visited, selects the node with greatest UTC value.
    # node	- node from which to select child node from.
    # -----------------------------------------------------------------------#

    @staticmethod
    def select_child(node: Node) -> Node:
        selected_child = node
        if node.num_children == 0:
            return node

        # check if any child were visited and return it
        for child in list(filter(None, list(node.children.values()))):
            if child.visits > 0.0:
                continue
            else:
                return child

        max_weight = 0.0
        for child in list(filter(None, list(node.children.values()))):
            # Weight = self.EvalUTC(Child)
            weight = child.sputc
            if weight >= max_weight:
                max_weight = weight
                selected_child = child
        return selected_child

    # -----------------------------------------------------------------------#
    # Description: Performs expansion phase of the MCTS.
    # Leaf	- Leaf node to expand.
    # -----------------------------------------------------------------------#
    def expansion_all_children(self, leaf):
        # if leaf.visits == 0 and leaf.parent is not None:
        #     return leaf
        # else:
        # Expand.
        if leaf.num_children == 0:
            self.eval_children(leaf, self.possible_actions)

        return self.select_expansion_action(leaf.state, self.possible_actions)

    def expansion_one_child(self, leaf):
        action = self.select_expansion_action(leaf, self.possible_actions)
        self.eval_children(leaf, [action])
        return action

    # -----------------------------------------------------------------------#
    # Description:
    #	Evaluates all the possible children states given a node state
    #	and returns the possible children Nodes.
    # node	- node from which to evaluate children.
    # -----------------------------------------------------------------------#
    def eval_children(self, node, actions):  # TODO - we must not simulate when picking a real action, search: # select next action
        # evaluate all possible next states
        for action in actions:
            _, alive_zombies = TreeAgent.simulate_action(node.state, self.agent_type, action)
            node.add_child(alive_zombies, action)

        return node.children

    # -----------------------------------------------------------------------#
    # Description:
    #	Selects a child node randomly.
    # node	- node from which to select a random child.
    # -----------------------------------------------------------------------#
    @staticmethod
    def select_expansion_action(node, possible_actions):
        # Wisely selects a child node.
        selected_child = TreeAgent.select_child(node)
        selected_action = None
        for key, value in node.children.items():
            if value == selected_child:
                selected_action = key
        if selected_action is None:
            selected_action = random.sample(possible_actions, 1)[0]

        return selected_action

    @staticmethod
    def select_simulation_action(alive_zombies, possible_actions):
        # Randomly selects a child node.
        i = random.sample(possible_actions, 1)[0]
        return i

    # -----------------------------------------------------------------------#
    # Description:
    #	Performs the simulation phase of the MCTS.
    #   for now, the agent takes random actions during the simulation
    # node	- node from which to perform simulation.
    # -----------------------------------------------------------------------#
    def simulation(self, expansion_action, selected_child):
        """
        Simulating states from previous states and actions
        This phase happens right after we've chose the expansion, and from the selected child with action
        :param expansion_action: the action from the selected_child
        :param selected_child: the child from the selection phase
        :return:
        """
        # this is here for enable the simulation to start from the root. by doing that we avoid the case of constant two steps at the beginning
        # we basically say that the root is the "selected child" and simulate from it, then the back-prop doesn't do much, updates only the root
        if selected_child.level <= self.simulation_depth:
            selected_child = self.root
            new_child = selected_child.add_child([], expansion_action, simulated_child=True)
            new_child.visits += 1
        # Perform simulation.
        total_reward = []
        for _ in range(self.simulation_num):
            simulation_reward = 0
            simulation_state = selected_child.children[expansion_action].state
            simulation_node = selected_child.children[expansion_action]
            for __ in range(self.simulation_depth):
                # select and execute next action (taken from simulation node)
                action = TreeAgent.select_simulation_action(simulation_state, self.possible_actions)
                one_step_reward, simulation_state = TreeAgent.simulate_action(simulation_state, self.agent_type, action)
                # create simulated child (without state - to prevent immediate-storage explosion)
                new_child = simulation_node.add_child([], action, simulated_child=True)
                new_child.visits += 1
                TreeAgent.EvalUTC(new_child, one_step_reward)  # evaluate UTC score of new child
                # set simulation node to current node
                simulation_node = new_child
                # aggregate reward
                simulation_reward += one_step_reward
                # print the tree
                # self.PrintTree()
            total_reward.append(simulation_reward)

        average_total_reward = np.average(total_reward) if self.agent_type == 'zombie' else -1 * np.average(total_reward)

        # back-prop from the expanded child (the child of the selected node)
        TreeAgent.back_propagation(selected_child.children[expansion_action], average_total_reward)

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
        CurrentNode = node

        # Update node's weight.
        TreeAgent.EvalUTC(CurrentNode, result)

        while TreeAgent.HasParent(CurrentNode):
            # Update parent node's weight.
            CurrentNode = CurrentNode.parent
            TreeAgent.EvalUTC(CurrentNode, result)

    @staticmethod
    def EvalUTC(node, result):
        if not node.simulated_node:
            node.wins += result
            node.ressq += result ** 2
            node.visits += 1

        UTC = node.wins / node.visits + TreeAgent.evaluate_exploration(node)

        node.sputc = UTC

    @staticmethod
    def evaluate_exploration(node):
        n = node.visits
        if node.parent is None:
            t = node.visits
        else:
            t = node.parent.visits

        # avoid log of 0 with: 't or 1'
        return C * np.sqrt(np.log(t or 1) / n)

    @staticmethod
    def HasParent(node):
        if node.parent is None:
            return False
        else:
            return True

    def reset(self):
        self.temporary_root = self.root
        # if self.agent_type == 'zombie':
        #     self.PrintTree()

    # -----------------------------------------------------------------------#
    # Description:
    #	Prints the tree to file.
    # -----------------------------------------------------------------------#
    def PrintTree(self):
        f = open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'Tree.txt'), 'w')
        node = self.root
        self.PrintNode(f, node, "")
        f.close()

    # -----------------------------------------------------------------------#
    # Description:
    #	Prints the tree node and its details to file.
    # node			- node to print.
    # Indent		- Indent character.
    # IsTerminal	- True: node is terminal. False: Otherwise.
    # -----------------------------------------------------------------------#
    def PrintNode(self, file, node, Indent):
        file.write(Indent)
        file.write("|-")
        Indent += "| "

        string = str(node.level) + " ("
        string += "W: " + str(node.wins) + ", N: " + str(node.visits) + ", UTC: " + str(node.sputc) + ") \n"
        file.write(string)

        for child in list(filter(None, list(node.children.values()))):
            self.PrintNode(file, child, Indent)
