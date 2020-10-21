import copy
import math
import os
import random
import multiprocessing as mp
from environment.game import Game
from agents.agent import Agent
from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy
from runnable_scripts.Utils import get_config
from core.node import Node
import numpy as np
from core.costly_simulation import CostlySimulation


def update_variables():
    MAX_HIT_POINTS = int(get_config("MainInfo")['max_hit_points'])
    MAX_ANGLE = int(get_config("MainInfo")['max_angle'])
    MAX_VELOCITY = int(get_config("MainInfo")['max_velocity'])
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
    C = float(get_config("TreeAgentInfo")['exploration_const'])
    return MAX_HIT_POINTS, MAX_ANGLE, MAX_VELOCITY, BOARD_WIDTH, BOARD_HEIGHT, C


class BasicMCTSAgent(Agent):
    MAX_HIT_POINTS = int(get_config("MainInfo")['max_hit_points'])
    MAX_ANGLE = int(get_config("MainInfo")['max_angle'])
    MAX_VELOCITY = int(get_config("MainInfo")['max_velocity'])
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
    C = float(get_config("TreeAgentInfo")['exploration_const'])

    def __init__(self, device, agent_type):
        BasicMCTSAgent.MAX_HIT_POINTS, BasicMCTSAgent.MAX_ANGLE, BasicMCTSAgent.MAX_VELOCITY, BasicMCTSAgent.BOARD_WIDTH, BasicMCTSAgent.BOARD_HEIGHT, BasicMCTSAgent.C = update_variables()
        super().__init__(EpsilonGreedyStrategy(), agent_type)
        self.possible_actions = list(range(Game.BOARD_HEIGHT)) if self.agent_type == 'zombie' else list(range(Game.BOARD_HEIGHT * Game.BOARD_WIDTH))
        self.root = Node([], self.possible_actions)
        self.temporary_root = self.root  # TODO - change its name to something like: real world state-node
        self.current_step = 0
        self.simulation_reward = 0
        self.simulation_num = int(get_config("TreeAgentInfo")['simulation_num'])  # number of simulations in the simulation phase
        self.simulation_depth = int(get_config("TreeAgentInfo")['simulation_depth'])  # number of times to expand a node in single simulation
        self.episode_reward = 0
        self.tree_depth = 0
        self.NUM_CORE = 12
        self.pool = mp.Pool(self.NUM_CORE)

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1

        # selection phase
        selected_child = self.selection()

        # expansion phase, here we selecting the action from which we will simulate the selected_child play-out
        # keep in mind that in this phase we expand a node that is NOT the temporary root, the expansion action doesn't relate to the real action we are taking
        # action = self.expansion_all_children(selected_child)
        expansion_action = self.expansion_one_child(selected_child)

        # simulation phase
        self.simulation(expansion_action, selected_child)

        # select next action
        action = self.select_expansion_action(self.temporary_root, self.possible_actions)
        self.eval_children(self.temporary_root, [action])
        self.temporary_root = self.temporary_root.children[action]
        # self.PrintTree()

        return action, rate, self.current_step

    def learn(self, _, action, __, reward):
        # back-propagation phase, start back-propagating from the current real world node
        # self.episode_reward += reward
        pass

    def selection(self):
        selected_child = self.temporary_root

        # Check if child nodes exist.
        if selected_child.num_children > 0:
            HasChild = True
        else:
            HasChild = False

        while HasChild:
            selected_child = self.select_child(selected_child)
            if selected_child.num_children == 0 or selected_child.is_terminal or selected_child.simulated_node:
                HasChild = False

        return selected_child

    # -----------------------------------------------------------------------#
    # Description:
    #	Given a node, selects the first unvisited child node, or if all
    # 	children are visited, selects the node with greatest UTC value.
    # node	- node from which to select child node from.
    # -----------------------------------------------------------------------#
    def select_child(self, node: Node) -> Node:
        selected_child = node
        if node.num_children == 0:
            return node

        # check if 'node' has any unexpanded nodes - which is any None value in children dictionary OR there is a child but it's simulated
        not_visited_actions = []
        for action, child in node.children.items():
            if child is None or child.simulated_node:
                # create the unexpanded child
                not_visited_actions.append(action)
        # chosen child from one of the unexpanded children - if there are any
        if len(not_visited_actions) > 0:
            action = random.sample(not_visited_actions, 1)[0]
            _, alive_zombies = BasicMCTSAgent.simulate_action(node.state, self.agent_type, action)
            return node.add_child(alive_zombies, action)

        max_weight = 0.0
        possible_children = []
        for child in list(node.children.values()):
            weight = child.uct
            if len(possible_children) == 0:
                possible_children.append(child)
                max_weight = weight
            elif weight == max_weight:
                possible_children.append(child)
            elif weight > max_weight:
                possible_children = [child]
                max_weight = weight
        if len(possible_children) > 0:
            selected_child = random.sample(possible_children, 1)[0]
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
            _, alive_zombies = BasicMCTSAgent.simulate_action(node.state, self.agent_type, action)
            node.add_child(alive_zombies, action)

        return node.children

    # -----------------------------------------------------------------------#
    # Description:
    #	Selects a child node randomly.
    # node	- node from which to select a random child.
    # -----------------------------------------------------------------------#
    def select_expansion_action(self, node, possible_actions):
        # Wisely selects a child node.
        selected_child = self.select_child(node)
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
        # if selected_child.level <= self.simulation_depth:
        #     selected_child = self.root
        #     new_child = selected_child.add_child([], expansion_action, simulated_child=True)
        #     new_child.visits += 1
        # Perform simulation.
        list_of_objects = []
        simulation_state = selected_child.children[expansion_action].state

        for _ in range(self.simulation_num):
            obj = CostlySimulation(self.simulation_depth, simulation_state, self.possible_actions, self.agent_type)
            list_of_objects.append(obj)

        list_of_results = self.pool.map(BasicMCTSAgent.worker, ((obj, BasicMCTSAgent.BOARD_HEIGHT, BasicMCTSAgent.BOARD_WIDTH) for obj in list_of_objects))
        # pool.close()
        # pool.join()

        average_total_reward = np.average(list_of_results) if self.agent_type == 'zombie' else -1 * np.average(list_of_results)

        # back-prop from the expanded child (the child of the selected node)
        BasicMCTSAgent.back_propagation(selected_child.children[expansion_action], average_total_reward)

    @staticmethod
    def worker(arg):
        return arg[0].costly_simulation(arg[1], arg[2])

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
            light_action = np.random.randint(0, BasicMCTSAgent.BOARD_HEIGHT * BasicMCTSAgent.BOARD_WIDTH)
        else:
            light_action = action
            # sample n times from zombie-agent actions-space
            zombie_action = np.random.randint(0, BasicMCTSAgent.BOARD_HEIGHT)

        # simulate and aggregate reward
        total_reward = 0
        new_zombie = Game.create_zombie(zombie_action)
        alive_zombies.append(new_zombie)
        reward, alive_zombies = Game.calc_reward_and_move_zombies(alive_zombies, light_action)
        total_reward += reward

        return total_reward, alive_zombies

    @staticmethod
    def back_propagation(node, result):
        result = result  # / (int(get_config('MainInfo')['zombies_per_episode']) + 3)
        CurrentNode = node

        # Update node's weight.
        BasicMCTSAgent.EvalUTC(CurrentNode, result)

        while BasicMCTSAgent.HasParent(CurrentNode):
            # Update parent node's weight.
            CurrentNode = CurrentNode.parent
            BasicMCTSAgent.EvalUTC(CurrentNode, result)

    @staticmethod
    def EvalUTC(node, result):
        if not node.simulated_node:
            node.wins += result
            node.visits += 1

        UTC = node.wins / node.visits + BasicMCTSAgent.evaluate_exploration(node)

        node.uct = UTC

    @staticmethod
    def evaluate_exploration(node):
        n = node.visits
        if node.parent is None:
            t = node.visits
        else:
            t = node.parent.visits

        # avoid log of 0 with: 't or 1'
        return BasicMCTSAgent.C * np.sqrt(np.log(t or 1) / n)

    @staticmethod
    def HasParent(node):
        if node.parent is None:
            return False
        else:
            return True

    def reset(self):
        # BasicMCTSAgent.back_propagation(self.temporary_root, self.episode_reward)
        self.temporary_root = self.root
        # self.episode_reward = 0
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
        string += "W: " + str(node.wins) + ", N: " + str(node.visits) + ", UCT: " + str(node.uct) + ") \n"
        file.write(string)

        for child in list(filter(None, list(node.children.values()))):
            self.PrintNode(file, child, Indent)
