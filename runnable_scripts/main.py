import os
import time
import torch
import matplotlib.pyplot as plt
from environment.game import Game
from agents.ddqn_agent import DdqnAgent
from runnable_scripts.Utils import create_dir, ridge_plot, save_ini_file


def main():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    plt.style.use('dark_background')
    path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "results",
                        time.strftime('%Y_%m_%d') + "_at_" + time.strftime('%H_%M'))
    create_dir(path)

    # create the game with the required agents
    env = Game(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), agent_zombie=DdqnAgent, agent_light=DdqnAgent)
    # play the game to produce the dictionaries including the results
    episodes_dict, steps_dict_light, steps_dict_zombie = env.play_game(path)

    # save and create results graph
    results_file_name = '/results_' + time.strftime('%d_%m_%Y_%H_%M')
    save_ini_file(path, results_file_name, steps_dict_light, steps_dict_zombie, episodes_dict)
    ridge_plot(path, results_file_name + '.xlsx')

    print('eliav king')


if __name__ == "__main__":
    main()

    """        
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
