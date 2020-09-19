import os
import time
import torch
import numpy as np
import pandas as pd
from IPython import display
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import ticker
from numpy.random import RandomState
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

from core.experience import Experience


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


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


def plot(is_ipython, values, moving_avg_period):
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
    if is_ipython:
        display.clear_output(wait=True)
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
        print('eliav king')


def save_check_point(dir, episode, episodes_dict, is_ipython, optimizer_light, optimizer_zombie, policy_net_light, policy_net_zombie, target_net_light,
                     target_net_zombie, CHECKPOINT):
    save_checkpoint(episode, target_net_zombie, policy_net_zombie, optimizer_zombie, 0,
                    dir + '/zombie.pth')
    save_checkpoint(episode, target_net_light, policy_net_light, optimizer_light, 0,
                    dir + '/light.pth')
    fig = plot(is_ipython, episodes_dict['episode_rewards'], CHECKPOINT)
    plt.savefig(dir + '/reward.png', bbox_inches='tight')
    plt.close(fig)
    df = pd.DataFrame({'reward': list(torch.cat(episodes_dict['episode_rewards'], -1).numpy()), 'episode_duration': episodes_dict['episode_durations']})
    df.to_csv(dir + '/log.csv')


def ridge_plot(dir_path, xlsx_name, values_per_column):
    sheets = ['light_actions', 'zombie_actions']
    x_start = 1
    y_start = 0

    for sheet in sheets:
        df = pd.read_excel(dir_path + xlsx_name, sheet_name=sheet)
        data = pd.DataFrame(
            data=np.transpose(
                np.array([list(map(lambda x: x // values_per_column, np.array(list(df.index)))), np.array(df['action']), np.ones(len(df))])),
            columns=['step', 'action', 'sum']).groupby(['step', 'action']).sum().reset_index()

        j = 1
        flag = True
        while j < len(data):
            expected_value = np.mod(1 + data['action'][j - 1], len(np.unique(data['action'])))
            max_action = max(data['action'])
            # check if row is missing! - we need to fill the next row with values
            if data['action'][j] != expected_value:
                data = pd.concat([data.iloc[0:j, :],
                                  pd.DataFrame([data['step'][j - 1] if expected_value == max_action else data['step'][j], expected_value, 0],
                                               index=data.columns.values, columns=[j]).T, data.iloc[j:, :]])
                j -= 1
                data.reset_index(drop=True, inplace=True)
            j += 1
            # at the end of the loop, we need to verify the last step series has the exact number of actions
            if j == len(data) and data['action'][j - 1] != max_action and flag:
                data = data.append(pd.DataFrame([data['step'][j - 1], max_action, 0], index=data.columns.values, columns=[j]).T)
                data.reset_index(drop=True, inplace=True)
                flag = False

        steps = [x for x in np.unique(data.step)]
        colors = rgb_generator(len(steps))

        gs = grid_spec.GridSpec(len(steps), 1)  # chagne to 4 columns
        fig = plt.figure(figsize=(16, 9))
        sns.set_style("dark")

        ax_objs = []
        for i in range(len(steps)):
            step = steps[i]
            x = data[data['step'] == step]['sum']
            x_d = np.linspace(0, len(x), len(x))

            # creating new axes object
            ax_objs.append(fig.add_subplot(gs[i:i + 1, 0:]))  # change to 0:-1, the last column should contain candles

            # plotting the distribution
            ax_objs[-1].plot(x_d, x, color="#f0f0f0", lw=1)
            ax_objs[-1].fill_between(x_d, x, alpha=1, color=colors[i])

            # setting uniform x and y lims
            ax_objs[-1].set_xlim(x_start, len(x))
            ax_objs[-1].set_ylim(y_start, np.max(data['sum']))

            # make background transparent
            rect = ax_objs[-1].patch
            rect.set_alpha(0)

            # remove borders, axis ticks, and labels
            ax_objs[-1].set_yticklabels([])

            if i == len(steps) - 1:
                ax_objs[-1].set_xlabel("Actions", fontsize=16, fontweight="bold")
                plt.setp(plt.gcf().get_axes(), yticks=[])
            else:
                ax_objs[-1].set_xticklabels([])
                ax_objs[-1].axis("off")

            spines = ["top", "right", "left", "bottom"]
            for s in spines:
                ax_objs[-1].spines[s].set_visible(False)

            if i == len(steps) - 1:
                ax_objs[-1].text(x_start - 0.02, y_start, 'Test Episodes: \n\n' + str(int(step * 100)) + ' - ' + str(int(step * 100 + 100)), fontweight="bold",
                                 fontsize=14, ha="right")
            elif i == 0:
                ax_objs[-1].text(x_start - 0.02, y_start, 'Episodes: \n\n' + str(int(step * 100)) + ' - ' + str(int(step * 100 + 100)) + '\n',
                                 fontweight="bold", fontsize=14, ha="right")
            else:
                ax_objs[-1].text(x_start - 0.02, y_start, str(int(step * 100)) + ' - ' + str(int(step * 100 + 100)) + '\n', fontweight="bold", fontsize=14,
                                 ha="right")
            # note: the ax.figbox is the figure box object with "min" attribute that stores the x's start,end values. We are interested in the first (start)
            ## ax_objs[-1].figbox.min[0]

        gs.update(hspace=-0.4)

        fig.text(0.18, 0.88, "Actions distribution along different ranges of episodes", fontsize=30)

        plt.savefig(dir_path + '\\' + sheet + '_ridge_plot.png', bbox_inches="tight")

        print('eliav king')


if __name__ == '__main__':
    temp = 2
    if temp == 1:
        dir_path = 'C:/Users/ELIAV/Google Drive/Final Project/FinalProjectRL/results/23_08_2020_at_08_18'
        xlsx_name = '/results_23_08_2020_13_13.xlsx'
        resolution = 20000
        STEPS_PER_EPISODE = 200
        eps_action_hist(dir_path, xlsx_name, resolution, STEPS_PER_EPISODE)
    elif temp == 2:
        dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + '\\results\\fixed_light_size_3_range_of_board_5_40\\2020_09_15_at_08_14'
        xlsx_name = '\\results_15_09_2020_10_50.xlsx'
        ridge_plot(dir_path=dir_path, xlsx_name=xlsx_name, values_per_column=10700)

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
"""
