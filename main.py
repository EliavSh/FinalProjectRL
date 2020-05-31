import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

from gameGrid import GameGrid
import lightMaster
import zombieMaster
from env import Env
from envManager import EnvManager
from epsilonGreedyStrategy import EpsilonGreedyStrategy
from zombieMaster import ZombieMaster
from lightMaster import LightMaster
from replayMemory import ReplayMemory
from DQN import DQN
from experience import Experience
from qValues import QValues

"""
# create env_manager
# create the two agents with initial params
# play the game:
    1. zombie master takes an action - places a zombie somewhere
    2. light master takes an action - places the light somewhere
    3. calculate rewards and let the agents learn from it
    4. the environment is taking one step of all zombies
"""
# set seed
np.random.seed(679)
random.seed(679)

# top parameters
target_update = 10
num_episodes = 1000
# in env we defined 50 steps per episode

# learning parameters
batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
memory_size = 100000
lr = 0.001

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Env(GameGrid(8, 16))

em = EnvManager(env, device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

agent_zombie = ZombieMaster(strategy, em.num_actions_available()[1], device)
policy_net_zombie = DQN(em.get_screen_height(), em.get_screen_width(), env.action_space()[1]).to(device)
target_net_zombie = DQN(em.get_screen_height(), em.get_screen_width(), env.action_space()[1]).to(device)
memory_zombie = ReplayMemory(memory_size)
target_net_zombie.load_state_dict(policy_net_zombie.state_dict())
target_net_zombie.eval()
optimizer_zombie = optim.Adam(params=policy_net_zombie.parameters(), lr=lr)

agent_light = LightMaster(strategy, em.num_actions_available()[0], device)
policy_net_light = DQN(em.get_screen_height(), em.get_screen_width(), env.action_space()[0]).to(device)
target_net_light = DQN(em.get_screen_height(), em.get_screen_width(), env.action_space()[0]).to(device)
memory_light = ReplayMemory(memory_size)
target_net_light.load_state_dict(policy_net_light.state_dict())
target_net_light.eval()
optimizer_light = optim.Adam(params=policy_net_light.parameters(), lr=lr)

episode_rewards = []


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
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Total Zombies Survived')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\n",
          moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython: display.clear_output(wait=True)


for episode in range(num_episodes):
    em.reset()
    state = em.get_state()
    zombie_master_reward = 0
# plt.figure() ;plt.imshow(state.squeeze(0).permute(1, 2, 0).cpu(), interpolation='none'); plt.title('Processed screen example'); plt.show()
    for time_step in count():
        action_zombie = agent_zombie.select_action(state, policy_net_zombie)
        action_light = agent_light.select_action(state, policy_net_light)

        reward = em.take_action(env.start_positions[action_zombie.numpy()], action_light.numpy())
        zombie_master_reward += reward
        next_state = em.get_state()

        memory_zombie.push(Experience(state, action_zombie, next_state, reward))
        memory_light.push(Experience(state, action_light, next_state, -reward))

        state = next_state

        if memory_zombie.can_provide_sample(batch_size):
            experiences_zombie = memory_zombie.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences_zombie)

            current_q_values = QValues.get_current(policy_net_zombie, states, actions)
            next_q_values = QValues.get_next(target_net_zombie, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer_zombie.zero_grad()
            loss.backward()
            optimizer_zombie.step()

            experiences_light = memory_light.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences_light)

            current_q_values = QValues.get_current(policy_net_light, states, actions)
            next_q_values = QValues.get_next(target_net_light, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer_light.zero_grad()
            loss.backward()
            optimizer_light.step()

        if em.done:  # if the episode is done, store it's reward and plot the moving average
            episode_rewards.append(zombie_master_reward)
            plot(episode_rewards, 10)
            break

    if episode % target_update == 0:
        target_net_light.load_state_dict(policy_net_light.state_dict())
        target_net_zombie.load_state_dict(policy_net_zombie.state_dict())

"""
# create the environment
env = env.Env(gameGrid.GameGrid(8, 16))
# create the agents
light_agent = lightMaster.LightMaster(env)
zombie_agent = zombieMaster.ZombieMaster(env)
game_window = env.Env(env)
for i in range(400):
    time.sleep(1)
    print("------ step", i, "------")
    zombie_action = zombie_agent.step()  # added zombie to env_manager
    light_action = light_agent.step()  # chose where to place the light
    reward = env.get_reward(light_action)
    zombie_agent.learn(reward)
    light_agent.learn(reward)
    game_window.update(env.alive_zombies, light_action)
    env.step(game_window.game_display.display.get_surface())

game_window.end_game()
"""
