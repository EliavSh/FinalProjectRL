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
from agents.alphaZero.utils import dotdict
from runnable_scripts.Utils import get_config
from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy
from core.neuralNets.NNet import NNetWrapper as nn

log = logging.getLogger(__name__)

args = dotdict({
    # 'numIters': 1000,
    # 'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    # 'tempThreshold': 15,  #
    'updateThreshold': 0.6,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    # 'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    # 'numMCTSSims': 5,  # Numbaer of games moves for MCTS to simulate.
    'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'train_examples_history': 200,

})


class AlphaZeroAgent(Agent):
    @staticmethod
    def update_variables():
        AlphaZeroAgent.BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
        AlphaZeroAgent.BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
        AlphaZeroAgent.LIGHT_SIZE = int(get_config("MainInfo")['light_size'])
        AlphaZeroAgent.DT = int(get_config("MainInfo")['dt'])
        AlphaZeroAgent.ANGLE = float(get_config("MainInfo")['max_angle'])
        AlphaZeroAgent.CPUCT = float(get_config('AlphaZeroInfo')['cpuct'])
        AlphaZeroAgent.TRAIN_EXAMPLES_HISTORY = float(get_config('AlphaZeroInfo')['train_examples_history'])
        return AlphaZeroAgent.BOARD_HEIGHT, AlphaZeroAgent.BOARD_WIDTH, AlphaZeroAgent.LIGHT_SIZE, AlphaZeroAgent.DT, AlphaZeroAgent.ANGLE, AlphaZeroAgent.CPUCT, AlphaZeroAgent.TRAIN_EXAMPLES_HISTORY

    # static field
    ZOMBIE_NUM = 1
    BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    LIGHT_SIZE = int(get_config("MainInfo")['light_size'])
    DT = int(get_config("MainInfo")['dt'])
    ANGLE = float(get_config("MainInfo")['max_angle'])
    CPUCT = float(get_config('AlphaZeroInfo')['cpuct'])
    TRAIN_EXAMPLES_HISTORY = float(get_config('AlphaZeroInfo')['train_examples_history'])

    def __init__(self, device, agent_type):
        super().__init__(EpsilonGreedyStrategy(), agent_type)
        AlphaZeroAgent.BOARD_HEIGHT, AlphaZeroAgent.BOARD_WIDTH, AlphaZeroAgent.LIGHT_SIZE, AlphaZeroAgent.DT, AlphaZeroAgent.ANGLE, AlphaZeroAgent.CPUCT, AlphaZeroAgent.TRAIN_EXAMPLES_HISTORY = AlphaZeroAgent.update_variables()
        args['cpuct'] = AlphaZeroAgent.CPUCT

        self.current_step = 0
        self.num_episode_per_learning = int(get_config("AlphaZeroInfo")['num_episode_per_learning'])
        self.current_episdoe = 0

        if agent_type == 'zombie':
            self.nnet = nn(AlphaZeroAgent.BOARD_WIDTH, AlphaZeroAgent.BOARD_HEIGHT, len(self.possible_actions))
        else:
            self.nnet = nn(AlphaZeroAgent.BOARD_WIDTH, AlphaZeroAgent.BOARD_HEIGHT * 2, len(self.possible_actions))
        self.pnet = self.nnet
        self.mcts = MCTS(self.nnet, self.possible_actions, self.agent_type, args)
        self.pi = 0
        self.train_examples = []
        self.train_examples_history = []

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1

        self.pi = self.mcts.getActionProb(state)

        return random.choice([i for i, v in enumerate(self.pi) if v == max(self.pi)]), rate, self.current_step
        # return np.random.choice(len(self.pi), p=self.pi), rate, self.current_step

    def learn(self, state, action, next_state, reward):
        # not really learning, just recording sample for later
        self.train_examples.append([state, self.pi, reward])

    def reset(self):
        self.train_examples_history.append(deepcopy(self.train_examples))
        self.train_examples = []
        self.current_episdoe += 1
        if len(self.train_examples_history) > AlphaZeroAgent.TRAIN_EXAMPLES_HISTORY:
            log.debug(
                f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.train_examples_history)}")
            self.train_examples_history.pop(0)

        if self.current_episdoe % self.num_episode_per_learning == 0:
            # once every 'something' episodes
            self.saveTrainExamples(self.current_episdoe)

            # shuffle examples before training
            trainExamples = []
            for e in self.train_examples_history:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.pnet, self.possible_actions, self.agent_type, args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.nnet, self.possible_actions, self.agent_type, args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.possible_actions, self.agent_type)
            pwins, nwins = arena.playGames(args.arenaCompare, self.agent_type)

            log.info('NEW/PREV WINS : %d / %d' % (nwins, pwins))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=args.checkpoint, filename=self.getCheckpointFile(self.current_episdoe))
                self.nnet.save_checkpoint(folder=args.checkpoint, filename='best.pth.tar')

        self.mcts = MCTS(self.nnet, self.possible_actions, self.agent_type, args)  # initiate the mcts for next episode

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        f = open(filename, "wb+")
        Pickler(f).dump(self.train_examples_history)

    def loadTrainExamples(self):
        modelFile = os.path.join(args.load_folder_file[0], args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            f = open(examplesFile, "rb")
            self.train_examples_history = Unpickler(f).load()
            log.info('Loading done!')
