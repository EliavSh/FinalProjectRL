import logging
import os
import random
import sys
from pickle import Pickler, Unpickler
from copy import deepcopy

import numpy as np
from random import shuffle

from agents.agent import Agent
from agents.alphaZero.Arena import Arena
from agents.alphaZero.MCTS import MCTS
from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy
from core.neuralNets.NNet import NNetWrapper

log = logging.getLogger(__name__)


class AlphaZeroAgent(Agent):
    def __init__(self, device, agent_type, config):
        super().__init__(agent_type, config)
        self.current_episode = 0

        # load values from config
        self.alpha_zero_info = config['AlphaZeroInfo']
        self.num_episode_per_learning = int(self.alpha_zero_info['num_episode_per_learning'])
        self.max_history_examples = int(self.alpha_zero_info['train_examples_history'])
        self.cpuct = float(self.alpha_zero_info['cpuct'])
        self.update_threshold = float(self.alpha_zero_info['update_threshold'])
        self.arena_compare = int(self.alpha_zero_info['arena_compare'])
        self.checkpoint = os.path.join(self.alpha_zero_info['checkpoint'], self.agent_type + "_player", self.__class__.__name__,
                                       "board_" + str(self.board_height) + "_" + str(self.board_width))
        self.load_model = eval(self.alpha_zero_info['load_model'])

        if agent_type == 'zombie':
            self.nnet = NNetWrapper(self.board_width, self.board_height, len(self.possible_actions))
        else:
            self.nnet = NNetWrapper(self.board_width, self.board_height * 2, len(self.possible_actions))

        if self.load_model:
            self.nnet.load_checkpoint(folder=self.checkpoint, filename='best.pth.tar')

        self.pnet = self.nnet
        self.mcts = MCTS(self.nnet, self.possible_actions, self.agent_type, self.alpha_zero_info, self.board_height, self.board_width, self.heal_points,
                         self.light_size, self.max_hit_points)
        self.pi = 0
        self.train_examples = []
        self.train_examples_history = []

    def get_neural_network(self):
        return self.nnet.nnet

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1

        if self.current_step < self.end_learning_step:
            self.pi = self.mcts.getActionProb(state)
            return np.random.choice(len(self.pi), p=self.pi), rate, self.current_step
        else:
            # Test phase, performs argmax of policy prediction
            self.pi, _ = self.nnet.predict(state)
            return random.choice([i for i, v in enumerate(self.pi) if v == max(self.pi)]), rate, self.current_step

    def learn(self, state, action, next_state, reward):
        # not really learning, just recording sample for later
        self.train_examples.append([state, self.pi, reward])

    def reset(self):
        self.train_examples_history.append(deepcopy(self.train_examples))
        self.train_examples = []
        self.current_episode += 1
        if len(self.train_examples_history) > self.max_history_examples:
            log.debug(
                f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.train_examples_history)}")
            self.train_examples_history.pop(0)

        if self.current_episode % self.num_episode_per_learning == 0 and self.current_step < self.end_learning_step:
            # once every 'something' episodes

            # shuffle examples before training
            trainExamples = []
            for e in self.train_examples_history:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.pnet, self.possible_actions, self.agent_type, self.alpha_zero_info, self.board_height, self.board_width, self.heal_points,
                         self.light_size, self.max_hit_points)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.nnet, self.possible_actions, self.agent_type, self.alpha_zero_info, self.board_height, self.board_width, self.heal_points,
                         self.light_size, self.max_hit_points)

            # if self.current_step < self.end_learning_step:
            if self.current_step < -float('inf'):
                log.info('PITTING AGAINST PREVIOUS VERSION')
                arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                              lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.possible_actions,
                              self.agent_type, self.board_height, self.board_width, self.zombies_per_episode, self.heal_points, self.max_hit_points,
                              self.light_size)
                pwins, nwins = arena.playGames(self.arena_compare)

                log.info('NEW/PREV WINS : %d / %d' % (nwins, pwins))
                if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.update_threshold:
                    log.info('REJECTING NEW MODEL')
                    self.nnet.load_checkpoint(folder=self.checkpoint, filename='temp.pth.tar')
                else:
                    log.info('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.checkpoint,
                                              filename=self.getCheckpointFile(self.current_episode))
                    self.nnet.save_checkpoint(folder=self.checkpoint, filename='best.pth.tar')
            else:
                self.nnet.save_checkpoint(folder=self.checkpoint,
                                          filename=self.getCheckpointFile(self.current_episode))
                self.nnet.save_checkpoint(folder=self.checkpoint, filename='best.pth.tar')

        self.mcts = MCTS(self.nnet, self.possible_actions, self.agent_type, self.alpha_zero_info, self.board_height, self.board_width, self.heal_points,
                         self.light_size, self.max_hit_points)  # initiate the mcts for next episode

    @staticmethod
    def getCheckpointFile(iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'
