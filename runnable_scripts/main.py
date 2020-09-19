import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib
from itertools import count
import pandas as pd

from environment.gameGrid import GameGrid
from environment.env import Env
from environment.envManager import EnvManager
from core.epsilonGreedyStrategy import EpsilonGreedyStrategy
from entities.zombieMaster import ZombieMaster
from entities.lightMaster import LightMaster
from core.replayMemory import ReplayMemory
from core.DQN import DQN
from core.experience import Experience
from core.qValues import QValues
from runnable_scripts.Utils import extract_tensors, create_dir, eps_action_hist, save_check_point, ridge_plot

"""
# create env_manager
# create the two agents with initial params
# play the game:
    1. zombie master takes an action - places a zombie somewhere
    2. light master takes an action - places the light somewhere
    3. calculate rewards and let the agents learn from it
    4. the environment is taking one step of all zombies
"""
import cProfile
import os


def main(board_size):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "results",
                       time.strftime('%Y_%m_%d') + "_at_" + time.strftime('%H_%M'))
    create_dir(dir)

    # set seed
    np.random.seed(679)
    random.seed(679)

    # top parameters
    light_size = 3
    width = board_size
    height = board_size
    target_update = 10
    num_episodes = 900
    num_test_episodes = 100  # the amount of episodes with zero epsilon - only smart moves
    ZOMBIES_PER_EPISODE = 100
    STEPS_PER_EPISODE = ZOMBIES_PER_EPISODE + width + 1
    CHECKPOINT = 100
    values_per_column = (num_episodes + num_test_episodes) * (1 + STEPS_PER_EPISODE) / 10

    # learning parameters
    batch_size = 100
    gamma = 0.999
    eps_start = 1
    eps_end = 0.05
    eps_decay = 2 / (num_episodes * STEPS_PER_EPISODE)
    memory_size = 2000
    lr = 0.001

    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython: pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Env(GameGrid(height=height, width=width), STEPS_PER_EPISODE, light_size=light_size)
    em = EnvManager(env, device)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay, num_episodes * STEPS_PER_EPISODE)

    agent_zombie = ZombieMaster(strategy, em.num_actions_available()[1], device)
    policy_net_zombie = DQN(em.get_screen_height(), em.get_screen_width(), env.action_space()[1]).to(device)
    target_net_zombie = DQN(em.get_screen_height(), em.get_screen_width(), env.action_space()[1]).to(device)
    memory_zombie = ReplayMemory(memory_size)
    target_net_zombie.load_state_dict(policy_net_zombie.state_dict())
    target_net_zombie.eval()
    optimizer_zombie = optim.Adam(params=policy_net_zombie.parameters(), lr=lr)

    agent_light = LightMaster(strategy, em.num_actions_available()[0], device)
    policy_net_light = DQN(2 * em.get_screen_height(), em.get_screen_width(), env.action_space()[0]).to(device)
    target_net_light = DQN(2 * em.get_screen_height(), em.get_screen_width(), env.action_space()[0]).to(device)
    memory_light = ReplayMemory(memory_size)
    target_net_light.load_state_dict(policy_net_light.state_dict())
    target_net_light.eval()
    optimizer_light = optim.Adam(params=policy_net_light.parameters(), lr=lr)

    episodes_dict = {'episode_rewards': [], 'episode_durations': []}
    steps_dict_light = {'epsilon': [], 'action': [], 'step': []}
    steps_dict_zombie = {'epsilon': [], 'action': [], 'step': []}

    for episode in range(num_episodes + num_test_episodes):
        em.reset()
        state_zombie, state_light = em.get_state()
        zombie_master_reward = 0
        # plt.figure() ;plt.imshow(state.squeeze(0).permute(1, 2, 0).cpu(), interpolation='none'); plt.title('Processed screen example'); plt.show()
        episode_start_time = time.time()
        for time_step in count():
            action_zombie, rate = agent_zombie.select_action(state_zombie, policy_net_zombie)
            action_light = agent_light.select_action(state_light, policy_net_light)

            # update dict
            steps_dict_light['epsilon'].append(rate)
            steps_dict_light['action'].append(action_light.numpy()[0] // env.grid.get_width())
            steps_dict_light['step'].append(time_step)
            steps_dict_zombie['epsilon'].append(rate)
            steps_dict_zombie['action'].append(int(action_zombie.numpy()[0]))
            steps_dict_zombie['step'].append(time_step)

            reward = em.take_action(env.start_positions[action_zombie.numpy()], action_light.numpy())
            zombie_master_reward += reward
            next_state_zombie, next_state_light = em.get_state()

            memory_zombie.push(Experience(state_zombie.unsqueeze(0), action_zombie, next_state_zombie.unsqueeze(0), reward))
            memory_light.push(Experience(state_light.unsqueeze(0), action_light, next_state_light.unsqueeze(0), -reward))

            state_zombie, state_light = next_state_zombie, next_state_light

            if memory_light.can_provide_sample(batch_size):
                experiences_zombie = memory_zombie.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences_zombie)

                # In the mean time I'm trying to teach the light master alone (while the zombie master takes random actions)
                # so, all this part is surrounded with block comment
                current_q_values = QValues.get_current(policy_net_zombie, states, actions)
                next_q_values = QValues.get_next(target_net_zombie, next_states, policy_net_zombie)
                target_q_values = (next_q_values * gamma) + rewards

                loss_zombie = F.mse_loss(current_q_values, target_q_values)
                optimizer_zombie.zero_grad()
                loss_zombie.backward()
                optimizer_zombie.step()

                experiences_light = memory_light.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences_light)

                current_q_values = QValues.get_current(policy_net_light, states, actions)
                next_q_values = QValues.get_next(target_net_light, next_states, policy_net_light)
                target_q_values = (next_q_values * gamma) + rewards

                loss_light = F.mse_loss(current_q_values, target_q_values)
                optimizer_light.zero_grad()
                loss_light.backward()
                optimizer_light.step()

            if em.done:  # if the episode is done, store it's reward and plot the moving average
                episodes_dict['episode_rewards'].append(zombie_master_reward)
                episodes_dict['episode_durations'].append(time.time() - episode_start_time)
                break

        if episode % target_update == 0:
            # update the target net to the same model dict as the policy net
            target_net_light.load_state_dict(policy_net_light.state_dict())
            target_net_zombie.load_state_dict(policy_net_zombie.state_dict())

        if episode % CHECKPOINT == 0:
            save_check_point(dir, episode, episodes_dict, is_ipython, optimizer_light, optimizer_zombie, policy_net_light, policy_net_zombie, target_net_light,
                             target_net_zombie, CHECKPOINT)

    save_check_point(dir, num_episodes + num_test_episodes, episodes_dict, is_ipython, optimizer_light, optimizer_zombie, policy_net_light, policy_net_zombie,
                     target_net_light, target_net_zombie, CHECKPOINT)
    results_file_name = '/results_' + time.strftime('%d_%m_%Y_%H_%M')
    writer = pd.ExcelWriter(dir + results_file_name + '.xlsx')
    pd.DataFrame(np.transpose(np.array(list(steps_dict_light.values()))), columns=list(steps_dict_light.keys())).set_index('step').to_excel(writer,
                                                                                                                                            sheet_name='light_actions')
    pd.DataFrame(np.transpose(np.array(list(steps_dict_zombie.values()))), columns=list(steps_dict_zombie.keys())).set_index('step').to_excel(writer,
                                                                                                                                              sheet_name='zombie_actions')
    pd.DataFrame({'info': [target_update, num_episodes + num_test_episodes, STEPS_PER_EPISODE, CHECKPOINT, batch_size, gamma, eps_start, eps_end, eps_decay,
                           memory_size, lr,
                           light_size, target_net_zombie.__str__(), target_net_light.__str__()]},
                 index=['target_update', 'num_episodes', 'STEPS_PER_EPISODE', 'CHECKPOINT', 'batch_size', 'gamma', 'eps_start', 'eps_end', 'eps_decay',
                        'memory_size', 'lr', 'light_size', 'target_net_zombie', 'target_net_light']).to_excel(writer, sheet_name='info')
    pd.DataFrame({'reward': list(torch.cat(episodes_dict['episode_rewards'], -1).numpy()), 'episode_duration': episodes_dict['episode_durations']}).to_excel(
        writer, sheet_name='rewards summary')
    writer.save()
    eps_action_hist(dir, results_file_name + '.xlsx', values_per_column, STEPS_PER_EPISODE)
    ridge_plot(dir, results_file_name + '.xlsx', values_per_column)
    print('eliav king')


if __name__ == '__main__':
    for i in list(range(5, 41, 5)):
        main(i)

    # cProfile.run('main()')

    """
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    with PyCallGraph(output=GraphvizOutput()):
        cProfile.run('main()')
        
    plot the keep_alive functions:
        plot.plot(x,np.sin(np.pi*x/2),x,np.power(x,1/2),x,np.power(x,1/3),x,np.power(x,1/4),x,np.power(x,1/5))
        plot.plot(np.transpose([0.38]*100), np.linspace(0,1,100))
        plot.xlim((0,1))
        plot.ylim((0,1))
        plot.legend(['sin(x*pi/2)', 'x^(1/2)', 'x^(1/3)', 'x^(1/4)', 'x^(1/5)', 'x=0.9^9'], prop={'size': 30})
        
    plot replay memory rewards:
        import matplotlib.pyplot as plt
        plt.plot([exp.reward.numpy()[0] for exp in memory_light.memory])
        plt.show()        
    """
