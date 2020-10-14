import time
import torch

from environment.game import Game
from agents.constant_agent import ConstantAgent
from agents.ddqn_agent import DdqnAgent
from agents.tree_agent import TreeAgent
from agents.random_agent import RandomAgent
from runnable_scripts.Utils import create_dir, ridge_plot, save_ini_file


def main():
    # create directory for storing the results
    path = create_dir()

    # create the game with the required agents
    env = Game(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), agent_zombie=DdqnAgent, agent_light=DdqnAgent)

    # play the game and produce the dictionaries of the results
    episodes_dict, steps_dict_light, steps_dict_zombie = env.play_zero_sum_game(path)

    # save and create results graph
    results_file_name = '/results_' + time.strftime('%d_%m_%Y_%H_%M')
    save_ini_file(path, results_file_name, steps_dict_light, steps_dict_zombie, episodes_dict)
    ridge_plot(path, results_file_name + '.xlsx')

    print('eliav king')


if __name__ == "__main__":
    main()
