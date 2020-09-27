import os
import time
from configparser import ConfigParser

import torch
import numpy as np
import pandas as pd
from IPython import display
from numpy.random import RandomState
import seaborn as sns
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from matplotlib.patches import Patch

from core.experience import Experience


def save_ini_file(path, results_file_name, steps_dict_light, steps_dict_zombie, episodes_dict):
    writer = pd.ExcelWriter(path + results_file_name + '.xlsx')
    pd.DataFrame(np.transpose(np.array(list(steps_dict_light.values()))), columns=list(steps_dict_light.keys())).set_index('step').to_excel(writer,
                                                                                                                                            sheet_name='light_actions')
    pd.DataFrame(np.transpose(np.array(list(steps_dict_zombie.values()))), columns=list(steps_dict_zombie.keys())).set_index('step').to_excel(writer,
                                                                                                                                              sheet_name='zombie_actions')

    config_main = get_config('MainInfo')
    config_ddqn = get_config('MainInfo')
    config_strategy = get_config('MainInfo')

    pd.DataFrame(
        {'info': [config_ddqn['target_update'], int(config_main['num_episodes']) + int(config_main['num_test_episodes']), config_main['zombies_per_episode'],
                  config_main['check_point'], config_ddqn['batch_size'], config_ddqn['gamma'], config_strategy['eps_start'], config_strategy['eps_end'],
                  config_ddqn['memory_size'], config_ddqn['lr'], config_main['light_size']]},
        index=['target_update', 'total_episodes', 'zombies_per_episode', 'check_point', 'batch_size', 'gamma', 'eps_start', 'eps_end', 'memory_size', 'lr',
               'light_size']).to_excel(writer, sheet_name='info')

    pd.DataFrame({'reward': list(torch.cat(episodes_dict['episode_rewards'], -1).numpy()), 'episode_duration': episodes_dict['episode_durations']}).to_excel(
        writer, sheet_name='rewards summary')
    writer.save()


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def get_config(config_topic):
    config_object = ConfigParser()
    config_object.read('C:/Users/ELIAV/Google Drive/Final Project/FinalProjectRL/configs/config.ini')
    return config_object[config_topic]


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return t1, t2, t3, t4


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


def plot(values, moving_avg_period):
    # turn interactive mode off
    plt.ioff()

    fig = plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Total Zombies Survived')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    print("Episode", len(values), "\n",
          moving_avg_period, "episode moving avg:", moving_avg[-1])
    return fig


def save_checkpoint(episode, target_net, policy_net, optimizer, loss, path):
    torch.save({
        'episode': episode,
        'target_net_state_dict': target_net.state_dict(),
        'policy_net_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def rgb_generator(size):
    rgb_list = []
    for i in range(size):
        rgb_list.append(tuple((np.random.uniform(0.2, 0.4), np.random.uniform(0.4, 0.6), np.random.uniform(0.6, 0.8), np.random.uniform(0.2, 0.8))))
    return rgb_list


def eps_action_hist(dir_path, xlsx_name, values_per_column, STEPS_PER_EPISODE):
    sheets = ['light_actions', 'zombie_actions']
    for sheet in sheets:
        # load and set up the data frame
        output = pd.read_excel(dir_path + xlsx_name, sheet_name=sheet)

        data = pd.DataFrame(
            data=np.transpose(
                np.array([list(map(lambda x: x // values_per_column, np.array(list(output.index)))), np.array(output['action']), np.ones(len(output))])),
            columns=['step', 'action', 'sum']).groupby(['step', 'action']).sum().reset_index()
        rows = zip(data['step'] * values_per_column + values_per_column, data['action'], data['sum'])
        headers = ['step', 'action', 'sum']
        df = pd.DataFrame(rows, columns=headers)

        # define some properties: figsize, margins and colors
        fig, ax = plt.subplots(figsize=(12, 10))
        margin_bottom = np.zeros(len(df['step'].drop_duplicates()) - 1)

        # build the bar plot
        actions = df['action'].drop_duplicates()
        colors = rgb_generator(len(actions))

        for num, action in enumerate(actions):
            temp_df = df[df['action'] == action]
            # fill any place without value with zeros (its Negligible)
            for i in range(len(margin_bottom)):
                if not temp_df['step'].values.__contains__(values_per_column * (i + 1)):
                    temp_df = pd.concat(
                        [temp_df.iloc[0:i, :], pd.DataFrame([values_per_column * (i + 1), action, 0], index=temp_df.columns.values).T, temp_df.iloc[i:, :]])

            values = list(temp_df[temp_df['action'] == action].loc[:, 'sum'])
            mar_len = len(margin_bottom)  # length of margins, sometimes exceeds 10 - it's about not relevant residuals
            temp_df.iloc[0:mar_len, :].plot.bar(x='step', y='sum', ax=ax, stacked=True, bottom=margin_bottom, color=colors[num], label=num)
            margin_bottom += values[0:mar_len]

        # plt.show()
        # set the x-ticks as the episode value and other plot wrappers
        ax.set_xticklabels(
            list(range(int(values_per_column // STEPS_PER_EPISODE), int(1 + 10 * values_per_column // STEPS_PER_EPISODE),
                       int(values_per_column // STEPS_PER_EPISODE) or 1)),
            rotation=30)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Steps', fontsize=14)
        plt.title('Actions distribution along different ranges of episodes', fontsize=20)
        plt.tight_layout()

        plt.savefig(dir_path + '\\' + sheet + '_hist.png')
    print('finished plotting action hist')


def save_check_point(dir, episode, episodes_dict, optimizer_light, optimizer_zombie, policy_net_light, policy_net_zombie, target_net_light,
                     target_net_zombie, CHECKPOINT):
    save_checkpoint(episode, target_net_zombie, policy_net_zombie, optimizer_zombie, 0,
                    dir + '/zombie.pth')
    save_checkpoint(episode, target_net_light, policy_net_light, optimizer_light, 0,
                    dir + '/light.pth')
    fig = plot(episodes_dict['episode_rewards'], CHECKPOINT)
    plt.savefig(dir + '/reward.png', bbox_inches='tight')
    plt.close(fig)
    df = pd.DataFrame({'reward': list(torch.cat(episodes_dict['episode_rewards'], -1).numpy()), 'episode_duration': episodes_dict['episode_durations']})
    df.to_csv(dir + '/log.csv')


def plot_progress(path, episodes_dict, moving_average_period):
    fig = plot(episodes_dict['episode_rewards'], moving_average_period)
    plt.savefig(path + '/reward.png', bbox_inches='tight')
    plt.close(fig)
    df = pd.DataFrame({'reward': list(torch.cat(episodes_dict['episode_rewards'], -1).numpy()), 'episode_duration': episodes_dict['episode_durations']})
    df.to_csv(path + '/log.csv')


def ridge_plot(dir_path, xlsx_name):
    plt.style.use('dark_background')
    x_start = 1
    y_start = 0
    rewards = pd.read_csv(dir_path + '\\log.csv', index_col=0)
    num_episodes = rewards.shape[0]
    number_of_graphs = 10
    episodes_per_graph = int(num_episodes / number_of_graphs)
    rewards_min = np.min(rewards['reward'])
    rewards_max = np.max(rewards['reward'])

    light_data, zombie_data, num_of_learning_episodes = create_data_for_ultimate_plot(dir_path, xlsx_name, number_of_graphs, rewards.shape[0])

    steps_light = [x for x in np.unique(light_data.step)]
    steps_zombie = [x for x in np.unique(zombie_data.step)]

    gs = grid_spec.GridSpec(len(steps_light), 5)
    fig = plt.figure(figsize=(16, 9))

    test_episodes_start_flag = True

    ax_objs = []
    for i in range(0, len(steps_light)):
        step = steps_light[i]
        x_light = light_data[light_data['step'] == step]['sum']
        x_d_light = np.linspace(0, len(x_light), len(x_light))
        x_zombie = zombie_data[zombie_data['step'] == step]['sum']
        x_d_zombie = np.linspace(0, len(x_zombie), len(x_zombie))

        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[i:(i + 1), -1]))  # candle plot - spreads over two vertical cells in the subplot grid
        ax_objs.append(fig.add_subplot(gs[i:(i + 1), 0:-1]))  # ridge plots

        # plotting zombie distribution
        ax_objs[-1].plot(x_d_zombie, x_zombie, color="#f0f0f0", lw=1)
        ax_objs[-1].fill_between(x_d_zombie, x_zombie, alpha=1, color='firebrick')
        # plotting light distribution
        ax_objs[-1].plot(x_d_light, x_light, color="#f0f0f0", lw=1)
        ax_objs[-1].fill_between(x_d_light, x_light, alpha=1, color='mediumseagreen')
        # wider width of plots
        list(map(lambda x: x.set_lw(1.5), ax_objs[-1].lines))
        # plotting box-plot rewards distribution
        quote = rewards['reward'][i * episodes_per_graph:((i + 1) * episodes_per_graph)]
        bp = ax_objs[-2].boxplot(quote, vert=False, showfliers=False)
        colors = ['white', 'white', 'white', 'orange', 'white']
        elements = ['boxes', 'whiskers', 'means', 'medians', 'caps']
        for iterator in range(len(elements)):
            plt.setp(bp[elements[iterator]], color=colors[iterator])

        y_max = np.max([light_data[light_data['step'] == step]['sum'], zombie_data[zombie_data['step'] == step]['sum']])

        # setting uniform x and y lims - light
        ax_objs[-1].set_xlim(x_start, len(x_light))
        ax_objs[-1].set_ylim(y_start, y_max)
        # setting uniform x and y lims - rewards
        ax_objs[-2].set_xlim(rewards_min, rewards_max)
        ax_objs[-2].set_ylim(0.85, 1.3)

        spines = ["top", "right", "left", "bottom"]
        # make background transparent
        for axis_index in range(2):
            rect = ax_objs[-(axis_index + 1)].patch
            rect.set_alpha(0)
            # remove borders, axis ticks, and labels
            ax_objs[-(axis_index + 1)].set_yticklabels([])
            for s in spines:
                ax_objs[-(axis_index + 1)].spines[s].set_visible(False)

        def remove_x_axis():
            ax_objs[-1].set_xticklabels([])
            ax_objs[-1].axis("off")
            ax_objs[-2].set_xticklabels([])
            ax_objs[-2].axis("off")

        if i == 0:  # first row, starting with 'Episodes' header
            remove_x_axis()
            ax_objs[-1].text(x_start - len(x_zombie) / 100, y_start, 'Episodes: \n\n' + str(int(step * episodes_per_graph) + 1) + ' - ' + str(
                int(step * episodes_per_graph + episodes_per_graph)) + '\n', fontweight="bold", fontsize=14, ha="right", color='white')
        elif step * episodes_per_graph < num_of_learning_episodes:  # all the rows until the test episodes
            remove_x_axis()
            ax_objs[-1].text(x_start - len(x_zombie) / 100, y_start,
                             str(int(step * episodes_per_graph) + 1) + ' - ' + str(int(step * episodes_per_graph + episodes_per_graph)) + '\n',
                             fontweight="bold", fontsize=14, ha="right", color='white')
        elif i != len(steps_zombie) - 1 and test_episodes_start_flag:  # first test episode, starting with header
            test_episodes_start_flag = False
            remove_x_axis()
            ax_objs[-1].text(x_start - len(x_zombie) / 100, y_start,
                             'Test Episodes: \n\n' + str(int(step * episodes_per_graph) + 1) + ' - ' + str(
                                 int(step * episodes_per_graph + episodes_per_graph)) + '\n', fontweight="bold", fontsize=14, ha="right", color='white')
        elif i != len(steps_zombie) - 1:  # for all test episodes between the first and the last
            remove_x_axis()
            ax_objs[-1].text(x_start - len(x_zombie) / 100, y_start,
                             str(int(step * episodes_per_graph) + 1) + ' - ' + str(int(step * episodes_per_graph + episodes_per_graph)) + '\n',
                             fontweight="bold", fontsize=14, ha="right", color='white')
        else:  # finally, the last row of test episodes
            ax_objs[-1].set_xlabel("Actions", fontsize=16, fontweight="bold", color='white')
            plt.setp(plt.gcf().get_axes(), yticks=[])
            ax_objs[-2].set_xlabel("Zombies survived", fontsize=14, fontweight="bold")
            plt.setp(plt.gcf().get_axes(), yticks=[])
            ax_objs[-1].text(x_start - len(x_zombie) / 100, y_start,
                             str(int(step * episodes_per_graph) + 1) + ' - ' + str(int(step * episodes_per_graph + episodes_per_graph)) + '\n',
                             fontweight="bold", fontsize=14, ha="right", color='white')

    gs.update(hspace=-0.0)

    legend_elements = [Patch(facecolor='firebrick', label='Zombie'), Patch(facecolor='mediumseagreen', label='Light')]
    plt.figlegend(handles=legend_elements, loc='lower left')

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle("Actions and rewards distribution along different ranges of episodes", fontsize=30, color='white')

    plt.savefig(dir_path + '\\ultimate_ridge_box_plot.png', bbox_inches="tight")
    print('finished plotting action-reward ridge-box plots')


def create_data_for_ultimate_plot(dir_path, xlsx_name, number_of_graphs, num_of_episodes):
    sheets = ['light_actions', 'zombie_actions']
    datas = []
    df = 0
    for sheet in sheets:
        df = pd.read_excel(dir_path + xlsx_name, sheet_name=sheet)
        steps_per_range_of_episodes = df.shape[0] / number_of_graphs
        data = pd.DataFrame(
            data=np.transpose(
                np.array([list(map(lambda x: x // steps_per_range_of_episodes, np.array(list(df.index)))), np.array(df['action']), np.ones(len(df))])),
            columns=['step', 'action', 'sum']).groupby(['step', 'action']).sum().reset_index()

        j = 1
        flag = True
        while j < len(data):
            expected_value = np.mod(1 + data['action'][j - 1], len(np.unique(data['action'])))
            max_action = max(data['action'])
            # check if row is missing! - we need to fill the next row with values
            if data['action'][j] != expected_value:
                data = pd.concat([data.iloc[0:j, :],
                                  pd.DataFrame([data['step'][j] if expected_value == 0 else data['step'][j - 1], expected_value, 0],
                                               index=data.columns.values, columns=[j]).T, data.iloc[j:, :]])
                j -= 1
                data.reset_index(drop=True, inplace=True)
            j += 1
            # at the end of the loop, we need to verify the last step series has the exact number of actions
            if j == len(data) and data['action'][j - 1] != max_action and flag:
                data = data.append(pd.DataFrame([data['step'][j - 1], max_action, 0], index=data.columns.values, columns=[j]).T)
                data.reset_index(drop=True, inplace=True)
                flag = False
        datas.append(data)
    return datas[0], datas[1], df[df['epsilon'].diff() == 0].index[0] / (df.shape[0] / num_of_episodes)


if __name__ == '__main__':
    temp = 4
    if temp == 1:
        dir_path = 'C:/Users/ELIAV/Google Drive/Final Project/FinalProjectRL/results/23_08_2020_at_08_18'
        xlsx_name = '/results_23_08_2020_13_13.xlsx'
        resolution = 20000
        STEPS_PER_EPISODE = 200
        eps_action_hist(dir_path, xlsx_name, resolution, STEPS_PER_EPISODE)
    elif temp == 2:
        dir_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir)) + '\\results\\fixed_light_size_3_range_of_board_5_40\\2020_09_15_at_19_36'
        xlsx_name = '\\results_15_09_2020_22_54.xlsx'
        ridge_plot(dir_path=dir_path, xlsx_name=xlsx_name)
    elif temp == 3:
        dir_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir)) + '\\results\\2020_09_21_at_21_19'
        xlsx_name = '\\results_22_09_2020_03_20.xlsx'
        ridge_plot(dir_path=dir_path, xlsx_name=xlsx_name)
    elif temp == 4:
        dir_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir)) + '\\results\\fixed_light_size_3_range_of_board_5_30'
        for file in os.listdir(dir_path):
            for f in os.listdir(dir_path + "\\" + file):
                if f.endswith(".xlsx"):
                    ridge_plot(dir_path=dir_path + "\\" + file + "\\", xlsx_name=f)

    print('eliav king')

"""
# draw epsilon graph 
fig, ax = plt.subplots(figsize=(7, 5))
plt.plot(output['epsilon'])
plt.xlabel('Simulation steps', fontsize=14)
plt.ylabel('Epsilon', fontsize=14)
plt.title('Epsilon decrease over simulation steps')
plt.show()
plt.savefig(dir_path + '\\epsilon_graph.png')


from matplotlib.transforms import Bbox, TransformedBbox, BboxTransformTo

fig.axes[0].bbox = TransformedBbox(Bbox([[0.7663793103448276, 0.8147457627118644],[0.900001, 0.88]]), BboxTransformTo(
    TransformedBbox(Bbox([[0.0, 0.0], [16.0, 9.0]]), Affine2D([[0, 100., 0.], [100., 0., 0.], [0., 0., 1.]]))))

"""
