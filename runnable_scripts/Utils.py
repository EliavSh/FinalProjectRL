import os
import time
import torch
import numpy as np
import pandas as pd
from IPython import display
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import ticker

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
        rgb_list.append(tuple((np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8))))
    return rgb_list


def eps_action_hist(dir_path, xlsx_name, resolution, STEPS_PER_EPISODE):
    sheets = ['light_actions', 'zombie_actions']
    for sheet in sheets:
        # load and set up the data frame
        output = pd.read_excel(dir_path + xlsx_name, sheet_name=sheet)

        data = pd.DataFrame(
            data=np.transpose(np.array([list(map(lambda x: x // resolution, np.array(list(output.index)))), np.array(output['action']), np.ones(len(output))])),
            columns=['step', 'action', 'sum']).groupby(['step', 'action']).sum().reset_index()
        rows = zip(data['step'] * resolution + resolution, data['action'], data['sum'])
        headers = ['step', 'action', 'sum']
        df = pd.DataFrame(rows, columns=headers)

        # define some properties: figsize, margins and colors
        fig, ax = plt.subplots(figsize=(12, 10))
        margin_bottom = np.zeros(len(df['step'].drop_duplicates()) - 1)
        # colors = ["#006D2C", "#31A354", "#74C476"]  # TODO - change that to some general number of colors
        # build the bar plot
        actions = df['action'].drop_duplicates()
        colors = rgb_generator(len(actions))

        for num, action in enumerate(actions):
            values = list(df[df['action'] == action].loc[:, 'sum'])
            mar_len = len(margin_bottom)  # length of margins, sometimes exceeds 10 - it's about not relevant residuals
            df[df['action'] == action].iloc[0:mar_len, :].plot.bar(x='step', y='sum', ax=ax, stacked=True, bottom=margin_bottom, color=colors[num], label=num)
            margin_bottom += values[0:mar_len]

        # plt.show()
        # set the x-ticks as the episode value and other plot wrappers
        ax.set_xticklabels(
            list(range(int(resolution // STEPS_PER_EPISODE), int(1 + 10 * resolution // STEPS_PER_EPISODE), int(resolution // STEPS_PER_EPISODE) or 1)),
            rotation=30)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Steps', fontsize=14)
        plt.title('Actions distribution along different ranges of episodes', fontsize=20)
        plt.tight_layout()

        plt.savefig(dir_path + '\\' + sheet + '_hist.png')
        print('eliav king')


def save_check_point(dir, episode, episodes_dict, is_ipython, optimizer_light, optimizer_zombie, policy_net_light, policy_net_zombie, target_net_light,
                     target_net_zombie):
    save_checkpoint(episode, target_net_zombie, policy_net_zombie, optimizer_zombie, 0,
                    dir + '/zombie.pth')
    save_checkpoint(episode, target_net_light, policy_net_light, optimizer_light, 0,
                    dir + '/light.pth')
    fig = plot(is_ipython, episodes_dict['episode_rewards'], 20)
    plt.savefig(dir + '/reward.png', bbox_inches='tight')
    plt.close(fig)
    df = pd.DataFrame({'reward': list(torch.cat(episodes_dict['episode_rewards'], -1).numpy()), 'episode_duration': episodes_dict['episode_durations']})
    df.to_csv(dir + '/log.csv')


if __name__ == '__main__':
    dir_path = 'C:/Users/ELIAV/Google Drive/Final Project/FinalProjectRL/results/08_08_2020_at_01_06'
    xlsx_name = '/results_08_08_2020_04_41.xlsx'
    resolution = 20000
    STEPS_PER_EPISODE = 200
    eps_action_hist(dir_path, xlsx_name, resolution, STEPS_PER_EPISODE)

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
