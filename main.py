import env
import gameGrid
import lightMaster
import zombieMaster
import game
import time

"""
# create env
# create the two agents with initial params
# play the game:
    1. zombie master takes an action - places a zombie somewhere
    2. light master takes an action - places the light somewhere
    3. calculate rewards and let the agents learn from it
    4. the environment is taking one step of all zombies
"""

# create the environment
env = env.Env(gameGrid.GameGrid(8, 16))
# create the agents
light_agent = lightMaster.LightMaster(env)
zombie_agent = zombieMaster.ZombieMaster(env)
game_window = game.Game(env)
for i in range(400):
    time.sleep(1)
    print("------ step", i, "------")
    zombie_action = zombie_agent.step()  # added zombie to env
    light_action = light_agent.step()  # chose where to place the light
    reward = env.get_reward(light_action)
    zombie_agent.learn(reward)
    light_agent.learn(reward)
    game_window.update(env.alive_zombies, light_action)
    env.step()

game_window.end_game()
