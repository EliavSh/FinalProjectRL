import os
import shutil

dir_name = "DDQN_vs_AlphaZero"
dir_name_of_images = "reward"  # must be the name of one of the images

# create rewards dir
path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "results", dir_name, dir_name_of_images)
if not os.path.exists(path):
    os.mkdir(path)
    os.chmod(path, 777)

str_arr = []

for i in list(range(10)):
    name1 = 'DDQN'
    name2 = 'AlphaZero'
    str_arr.append(name1 + ' vs ' + name2 + ' on board 10 __ ' + str(i))
    str_arr.append(name1 + ' vs ' + name2 + ' on board 20 __ ' + str(i))
    str_arr.append(name1 + ' vs ' + name2 + ' on board 30 __ ' + str(i))
    str_arr.append(name2 + ' vs ' + name1 + ' on board 10 __ ' + str(i))
    str_arr.append(name2 + ' vs ' + name1 + ' on board 20 __ ' + str(i))
    str_arr.append(name2 + ' vs ' + name1 + ' on board 30 __ ' + str(i))

for root, dirs, files in os.walk(
        os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "results", dir_name)):
    if 'rewards' in dirs:
        dirs.remove('rewards')
    if len(files) > 0:
        shutil.copyfile(root + '\\' + files[files.index(dir_name_of_images + '.png')], path + '\\' + str_arr.pop(0) + ' .png')
