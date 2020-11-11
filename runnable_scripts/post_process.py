import os
import shutil

from numpy import linspace


dir_name = "mcts_vs_ddqn"

# create rewards dir
path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "results", dir_name, 'rewards')
if not os.path.exists(path):
    os.mkdir(path)
    os.chmod(path, 777)

str_arr = []

for size in range(2, 10, 2):
    for exploration_rate in linspace(0.0, 4., 21):
        str_arr.append('board_' + str(size) + '_exploration_' + str(round(exploration_rate, 2)))

for root, dirs, files in os.walk(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "results", dir_name)):
    if len(files) == 5:  # for the root, doesn't contains any files - only folders
        shutil.copyfile(root + '\\' + files[3], path + '\\' + str_arr.pop(0) + ' .png')
