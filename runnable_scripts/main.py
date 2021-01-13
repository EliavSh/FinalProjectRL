import os
import time
import torch
from numpy import linspace

from configparser import RawConfigParser

from core.node import Node
from core.zombie import Zombie
import logging


def main():
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "true"  # cancel py-game display
    from environment.game import Game
    from agents.random_agent import RandomAgent
    from agents.constant_agent import ConstantAgent
    from agents.ddqn_agent import DdqnAgent
    from agents.alphaZero.alpha_zero_agent import AlphaZeroAgent
    from runnable_scripts.Utils import create_dir, ridge_plot, save_ini_file

    # create directory for storing the results
    dir_path = create_dir()

    logging.basicConfig(filename=os.path.join(dir_path, 'logger.log'), filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    # create the game with the required agents
    env = Game(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), agent_zombie=ConstantAgent,
               agent_light=AlphaZeroAgent)

    # play the game and produce the dictionaries of the results
    episodes_dict, steps_dict_light, steps_dict_zombie = env.play_zero_sum_game(dir_path)

    # save and create results graph
    results_file_name = '/results_' + time.strftime('%d_%m_%Y_%H_%M')
    save_ini_file(dir_path, results_file_name, steps_dict_light, steps_dict_zombie, episodes_dict)
    ridge_plot(dir_path, results_file_name + '.xlsx')

    print('Eliav king')


if __name__ == "__main__":
    temp = 1
    if temp == 1:
        for board in range(20, 60, 10):
            # for cpuct in linspace(0.15, 1.65, 7):
            path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'configs',
                                'config.ini')
            parser = RawConfigParser()
            parser.read(path)
            parser.set('MainInfo', 'board_height', str(board))
            parser.set('MainInfo', 'board_width', str(board))
            config_file = open(path, 'w')
            parser.write(config_file, space_around_delimiters=True)
            config_file.close()

            # update all variables due to changes in the configuration file
            Zombie.update_variables()
            Node.update_variables()

            main()
    elif temp == 2:
        main()
