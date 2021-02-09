import os
import shutil

from numpy import linspace

dir_name = "Testing random vs max action choosing AND 3 vs 4 conv layers"

# create rewards dir
path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "results", dir_name, 'rewards')
if not os.path.exists(path):
    os.mkdir(path)
    os.chmod(path, 777)

str_arr = []

for cpuct in list(range(4)):
    str_arr.append('cpuct_' + str(cpuct))

for root, dirs, files in os.walk(
        os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "results", dir_name)):
    if len(files) == 6:  # for the root, doesn't contains any files - only folders
        shutil.copyfile(root + '\\' + files[4], path + '\\' + str_arr.pop(0) + ' .png')
