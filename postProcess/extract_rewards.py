import os
import shutil

dir_name = "alphazero_vs_const_double_gaussian_uniform"

# create rewards dir
path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "results", dir_name, 'rewards')
if not os.path.exists(path):
    os.mkdir(path)
    os.chmod(path, 777)

str_arr = []

for name in ['ConstantAgent', 'DoubleConstantAgent', 'GaussianAgent', 'UniformAgent']:
    str_arr.append('AlphaZero vs ' + name + ' on board 10')
    str_arr.append('AlphaZero vs ' + name + ' on board 20')
    str_arr.append('AlphaZero vs ' + name + ' on board 30')
    str_arr.append(name + ' vs AlphaZero' + ' on board 10')
    str_arr.append(name + ' vs AlphaZero' + ' on board 20')
    str_arr.append(name + ' vs AlphaZero' + ' on board 30')

for root, dirs, files in os.walk(
        os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "results", dir_name)):
    if 'rewards' in dirs:
        dirs.remove('rewards')
    if len(files) > 0:
        shutil.copyfile(root + '\\' + files[files.index('ultimate_ridge_box_plot.png')], path + '\\' + str_arr.pop(0) + ' .png')
