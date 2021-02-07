import os
import numpy as np
from configparser import ConfigParser
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_dir = os.path.abspath(os.pardir)
final_results_dir = os.path.join(os.path.abspath(os.pardir), 'final_results')
docs_dir = os.path.join(os.path.abspath(os.pardir), 'docs')


# get a list of final_results' directories
def get_list_dirs_relative_to_project_dir(path: str) -> List[str]:
    return os.listdir(os.path.join(project_dir, path))


# split a list directories to types, based on: "starts with ..." (DDQN / M.C.T.S / AlphaZero / neither)
def split_list_by_words_in_specific_position(list_of_scenarios: List[str], words: List[str], position: int) -> List[List[str]]:
    final_split_list = []
    scenarios_without_match = [False] * len(list_of_scenarios)
    for w in words:
        lst = list(map(lambda x: x if x.split('_')[position] == w else False, list_of_scenarios))
        scenarios_without_match = list(map(lambda x, y: x or y, scenarios_without_match, lst))
        final_split_list.append(list(filter(lambda x: x, lst)))  # we can also use: [x for x in lst if x]
    # adding the rest
    # final_split_list.append([y for x, y in zip(scenarios_without_match, list_of_scenarios) if not x])
    return final_split_list


# [[[DDQN vs Uniform , DDQN vs Uniform_1], [DDQN vs Gaussian, DDQN vs Gaussian_1]...], [[Uniform vs DDQN, Uniform_1 vs DDQN], [Gaussian vs DDQN, Gaussian_1 vs DDQN]...]]

# load rewards:
## for each inner list:
## init df of size (1000,27)
### for each value:
### validations: config has all values: board, target update, memory
#### load the df by mean: each column named after concatenation of board size, target update and memory size (from config)
# --#--#--# the output should be a flat list with a dict for each list with key of the name of the scenario and value of the df:
# --#--#--# ["ddqn_vs_uniform" : df, "ddqn_vs_gaussian" : df, ..., "uniform_vs_ddqn" : df, "gaussian_vs_ddqn" : df, ...]
def load_rewards(games_list: List[str], topic: str, tuning_parameters: List[str]) -> Dict[str, List]:
    final_scores_dict = {}
    for iterate, game in enumerate(games_list):
        for d in os.listdir(os.path.join(final_results_dir, game)):
            mean_test_reward, information = get_reward_from_scenario(os.path.join(final_results_dir, game, d), topic, tuning_parameters)
            key = str(list(information.items()))
            if key in final_scores_dict.keys():
                final_scores_dict[str(list(information.items()))] = [
                    (final_scores_dict[str(list(information.items()))][0] * iterate + mean_test_reward) / (iterate + 1), information]
            else:
                final_scores_dict[str(list(information.items()))] = [mean_test_reward, information]
    return final_scores_dict


def get_reward_from_scenario(path: str, topic: str, tuning_parameters: List[str]) -> [float, dict]:
    df = pd.read_csv(os.path.join(path, 'log.csv'))
    config_object = ConfigParser()
    config_object.read(os.path.join(path, 'config.ini'))
    board = int(config_object['MainInfo']['board_height'])
    info_dict = {'board': board}
    for param in tuning_parameters:
        info_dict[param] = config_object[topic][param]
    return df['reward'][int(config_object['MainInfo']['num_train_episodes']):].mean(), info_dict


# get_mean_test_score:
## for each inner dict:
### map the value to a list of dfs (as the number of game boards) in such way:
### each df consists of target_update (columns) and memory_size (rows) with value of mean_test_reward (3 by 3 in our case)

# print_tables
## for each list of dfs, plot tables in concatenated way (9 by 3 in our case. with separations of different board sizes)

def init_df_dict(parameters: Dict[str, List[int]]) -> Dict[int, pd.DataFrame]:
    df_dictionary = {}
    params_values = list(parameters.values())
    for board in [10, 20, 30]:
        df = pd.DataFrame(np.zeros((3, 3)), index=params_values[0], columns=params_values[1])
        df_dictionary[board] = df
    return df_dictionary


if __name__ == "__main__":
    learning_agent_topic = 'DdqnAgentInfo'
    boards = [10, 20, 30]
    params = {'memory_size': [3000, 4000, 5000], 'target_update': [500, 750, 1000]}
    df_dict = init_df_dict(params)

    scenarios_list = get_list_dirs_relative_to_project_dir('final_results')
    for agent_type in ['light', 'zombie']:
        ddqn_pos, dumb_pos = (0, 2) if agent_type == 'light' else (2, 0)
        all_games_of_agent = split_list_by_words_in_specific_position(list_of_scenarios=scenarios_list, words=['DDQN'], position=ddqn_pos)
        unique_games = split_list_by_words_in_specific_position(all_games_of_agent[0], ['Const', 'DoubleConst', 'Uniform', 'Gaussian'], dumb_pos)
        fig, axes = plt.subplots(len(boards), len(unique_games), figsize=(12, 9))
        for j, two_players_game in enumerate(unique_games):
            mean_test_score_dict = load_rewards(two_players_game, learning_agent_topic, list(params.keys()))
            # stream test mean values to df_dict
            for values in mean_test_score_dict.values():
                info = values[1]
                params_keys = list(params.keys())
                df_dict[info['board']].loc[int(info[params_keys[0]])][int(info[params_keys[1]])] = values[0]

            # plot heat-map for every df
            for i in range(len(boards)):
                sns.heatmap(df_dict[boards[i]], ax=axes[i][j], annot=True, fmt="0.2f")

                # set titles
                if i == 0:
                    axes[i][j].set_title(two_players_game[0])
                if j == 0:
                    axes[i][j].set_ylabel('Memory size')
                if i == len(boards) - 1:
                    axes[i][j].set_xlabel('Target update')

        for ax, row in zip(axes[:, 0], ['Board of {}'.format('\n' + str(board) + 'X' + str(board) + '  ') for board in boards]):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size=16, ha='right', va='center')

        plt.tight_layout()
        fig.savefig(os.path.join(docs_dir, 'DDQN_avg_test_rewards', 'DDQN as ' + agent_type + '.png'))
        print('king')

    print('king')
