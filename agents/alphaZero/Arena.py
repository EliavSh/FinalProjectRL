import numpy as np
import copy
import logging
from environment.game import Game
from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, possible_actions, agent_type, board_height, board_width, zombies_per_episode, heal_points, max_hit_points, light_size):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
        """
        self.player1 = player1
        self.player2 = player2
        self.possible_actions = possible_actions
        self.agent_type = agent_type
        self.board_height = board_height
        self.board_width = board_width
        self.zombies_per_episode = zombies_per_episode
        self.heal_points = heal_points
        self.max_hit_points = max_hit_points
        self.light_size = light_size

    def playGame(self, player):
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
        while it < self.zombies_per_episode + self.board_width:
            it += 1
            action = player(board)

            valids = np.ones((len(self.possible_actions)), dtype=int)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, reward = Game.get_next_state(board, self.agent_type, action, self.board_height, self.board_width, self.heal_points, self.max_hit_points, self.light_size)
            total_reward += reward
        return total_reward

    def playGames(self, num):
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
            one_rewards.append(self.playGame(self.player1))
        log.info(f'first player rewards: {one_rewards}')

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            two_rewards.append(self.playGame(self.player2))
        log.info(f'second player rewards: {two_rewards}')

        return self.get_total_wins(one_rewards, two_rewards), self.get_total_wins(two_rewards, one_rewards)

    @staticmethod
    def get_total_wins(rewards1, rewards2):
        return sum(list(map(lambda x, y: x > y, rewards1, rewards2)))

    def get_starting_state(self):
        zombie_grid = np.reshape(np.array(range(self.board_height * self.board_width)),
                                 [self.board_height, self.board_width])
        zombie_grid = zombie_grid.astype(np.float32)
        zombie_grid.fill(0)
        health_grid = copy.deepcopy(zombie_grid)
        return zombie_grid, np.concatenate((zombie_grid, health_grid))
