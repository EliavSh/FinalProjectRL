import os
import time
import torch
from configparser import RawConfigParser
import logging
from agents import *


def main(l_agent, z_agent):
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "true"  # cancel py-game display
    from environment.game import Game
    from runnable_scripts.Utils import create_dir, ridge_plot, save_ini_file

    # create directory for storing the results
    dir_path = create_dir()

    logging.basicConfig(filename=os.path.join(dir_path, 'logger.log'), filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

    # create the game with the required agents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Game(device, light_agent=l_agent, zombie_agent=z_agent)

    # play the game and produce the dictionaries of the results
    episodes_dict, steps_dict_light, steps_dict_zombie = env.play_zero_sum_game(dir_path)

    # save and create results graph
    results_file_name = 'results_' + time.strftime('%d_%m_%Y_%H_%M')
    save_ini_file(dir_path, results_file_name, steps_dict_light, steps_dict_zombie, episodes_dict)
    ridge_plot(dir_path, results_file_name + '.xlsx')

    print('Eliav king')


if __name__ == "__main__":
    temp = 1
    smart_agent = 'zombie'
    light_agent = UniformAgent
    zombie_agent = GaussianAgent
    if temp == 1:
        for iteration in [1, 2]:
            for board in range(10, 31, 10):
                for simulation_num in [10, 20, 30]:
                    for simulation_depth in [2, 3, 4]:
                        path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'configs', 'config.ini')
                        parser = RawConfigParser()
                        parser.read(path)
                        parser.set('MainInfo', 'board_height', str(board))
                        parser.set('MainInfo', 'board_width', str(board))
                        if smart_agent == 'zombie':
                            parser.set('MainInfo', 'light_size', str(board // 3))
                        else:
                            parser.set('MainInfo', 'light_size', str(2))
                        parser.set('TreeAgentInfo', 'simulation_num', str(simulation_num))
                        parser.set('TreeAgentInfo', 'simulation_depth', str(simulation_depth))
                        config_file = open(path, 'w')
                        parser.write(config_file, space_around_delimiters=True)
                        config_file.close()

                        main(light_agent, zombie_agent)
    elif temp == 2:
        main(light_agent, zombie_agent)
