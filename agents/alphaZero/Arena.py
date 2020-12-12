import numpy as np
import copy
import logging

from environment.game import Game
from tqdm import tqdm

from runnable_scripts.Utils import get_config

log = logging.getLogger(__name__)


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, possible_actions, agent_type):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.possible_actions = possible_actions
        self.agent_type = agent_type
        self.main_info = get_config("MainInfo")

    def playGame(self, player, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        board = self.get_starting_state()[0] if self.agent_type == 'zombie' else self.get_starting_state()[1]
        it = 0
        total_reward = 0
        while it < int(self.main_info['zombies_per_episode']) + int(self.main_info['board_width']):
            it += 1
            action = player(board)

            valids = np.ones((len(self.possible_actions)), dtype=int)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, reward = Game.get_next_state(board, self.agent_type, action)
            total_reward += reward
        return total_reward

    def playGames(self, num, agent_type, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
        """

        num = int(num / 2)
        one_rewards = []
        two_rewards = []
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            one_rewards.append(self.playGame(self.player1, verbose=verbose))
        log.info(f'first player rewards: {one_rewards}')

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            two_rewards.append(self.playGame(self.player2, verbose=verbose))
        log.info(f'second player rewards: {two_rewards}')

        return self.get_total_wins(one_rewards, two_rewards), self.get_total_wins(two_rewards, one_rewards)

    @staticmethod
    def get_total_wins(rewards1, rewards2):
        return sum(list(map(lambda x, y: x > y, rewards1, rewards2)))

    @staticmethod
    def get_starting_state():
        zombie_grid = np.reshape(np.array(range(Game.BOARD_HEIGHT * Game.BOARD_WIDTH)),
                                 [Game.BOARD_HEIGHT, Game.BOARD_WIDTH])
        zombie_grid = zombie_grid.astype(np.float32)
        zombie_grid.fill(0)
        health_grid = copy.deepcopy(zombie_grid)
        return zombie_grid, np.concatenate((zombie_grid, health_grid))
