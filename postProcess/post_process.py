import os
import numpy as np
from configparser import ConfigParser
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import pandas as pd

from runnable_scripts import Utils


class PostProcess:
    def __init__(self, smart_agent: str, learning_agent_topic: str, boards: List[int], params: Dict[str, List[int]], period: int):
        self.moving_average_period = period
        self.smart_agent = smart_agent
        self.learning_agent_topic = learning_agent_topic
        self.boards = boards
        self.params = params
        self.params_types = list(map(lambda x: type(x[0]), list(self.params.values())))
        self.project_dir = os.path.abspath(os.pardir)
        self.final_results_dir = os.path.join(os.path.abspath(os.pardir), 'final_results')
        self.text_color = 'w'
        self.agents_types = ['Single Action Agent', 'Double Action Agent', 'Uniform Agent', 'Gaussian Agent']
        self.score_param = 'Average Test Reward'
        self.docs_dir = os.path.join(os.path.abspath(os.pardir), 'docs')
        self.zombies_per_episode = 0  # to be filled
        self.fig = 'To be filled'
        self.axes = 'To be filled'

    # get a list of final_results' directories
    def get_list_dirs_relative_to_project_dir(self, path: str) -> List[str]:
        return os.listdir(os.path.join(self.project_dir, path))

    def load_rewards(self, games_list: List[str], player_type: str, rewards_of_all_agents: Dict[str, Dict[str, Any]], agent_type_index: int) \
            -> Tuple[Dict[str, List], Dict[str, Dict[str, Any]]]:
        topic = self.learning_agent_topic
        tuning_parameters = list(self.params.keys())
        final_scores_dict = {}
        for i, game in enumerate(games_list):
            for d in sorted(os.listdir(os.path.join(self.final_results_dir, game))):
                rewards, information, num_train_episodes = self.get_reward_from_scenario(os.path.join(self.final_results_dir, game, d), topic,
                                                                                         tuning_parameters, player_type)
                # save rewards for later
                rewards_of_all_agents[list(rewards_of_all_agents.keys())[agent_type_index]][game].append(rewards.values)

                # calculate mean test reward of the specific game
                mean_test_reward = rewards[num_train_episodes:].mean()
                key = str(list(information.items()))

                # average results in cases there are more than 1 game of the specific scenario
                if key in final_scores_dict.keys():
                    final_scores_dict[str(list(information.items()))] = [
                        (final_scores_dict[str(list(information.items()))][0] * i + mean_test_reward) / (i + 1), information]
                else:
                    final_scores_dict[str(list(information.items()))] = [mean_test_reward, information]

        rewards_of_all_agents[list(rewards_of_all_agents.keys())[agent_type_index]]['info'] = list(map(lambda x: x[1][1], [*final_scores_dict.items()]))
        return final_scores_dict, rewards_of_all_agents

    def get_reward_from_scenario(self, path: str, topic: str, tuning_parameters: List[str], player_type: str) -> [pd.Series, dict, int]:
        # load conf and set the value of zombies per episode
        conf = ConfigParser()
        conf.read(os.path.join(path, 'config.ini'))
        self.zombies_per_episode = int(conf['MainInfo']['zombies_per_episode'])

        df = pd.read_csv(os.path.join(path, 'log.csv'))
        board = int(conf['MainInfo']['board_height'])
        info_dict = {'board': board}
        for param in tuning_parameters:
            info_dict[param] = conf[topic][param]
        # the players get opposite rewards
        multiplier = 1 if player_type == 'zombie' else -1
        return df['reward'] * multiplier, info_dict, int(conf['MainInfo']['num_train_episodes'])

    def init_results_dicts(self) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
        df_dict = self.init_results_dict(self.params, self.boards)
        best_light_results = self.init_results_dict({'agent_type': self.agents_types, 'parameters': [*list(self.params.keys()), self.score_param]}, self.boards)
        return df_dict, best_light_results

    @staticmethod
    def init_results_dict(parameters, board_list):
        df_dictionary = {}
        params_values = list(parameters.values())
        for board in board_list:
            df = pd.DataFrame(np.zeros((len(params_values[0]), len(params_values[1]))), index=params_values[0], columns=params_values[1])
            df_dictionary[board] = df
        return df_dictionary

    def create_figure(self, num_row, num_col: int, width: int, height: int, header: str, font_size: int):
        self.fig, self.axes = plt.subplots(num_row, num_col, figsize=(width, height))
        self.fig.suptitle(header, fontsize=font_size, color=self.text_color)
        return self.fig, self.axes

    @staticmethod
    def upper_first_letter(string: str):
        return string.upper()[0] + string.lower()[1:]

    def set_subplots_title(self, axes, two_players_game, i, j):
        params_keys = list(self.params.keys())
        # set titles
        if i == 0:
            axes[i][j].set_title(self.name_conversion(two_players_game[0]), color=self.text_color)
            axes[i][j].title.set_color(self.text_color)
        if j == 0:
            axes[i][j].set_ylabel(self.upper_first_letter(params_keys[0].split('_')[0]) + ' ' + self.upper_first_letter(params_keys[0].split('_')[1]),
                                  color=self.text_color, fontsize=14)
            axes[i][j].set_yticklabels(axes[i][j].get_yticklabels(), verticalalignment='center')
        if i == len(self.boards) - 1:
            axes[i][j].set_xlabel(self.upper_first_letter(params_keys[1].split('_')[0]) + ' ' + self.upper_first_letter(params_keys[1].split('_')[1]),
                                  color=self.text_color, fontsize=14)
        # remove x and y ticks of inner subplots
        if i != len(self.boards) - 1:
            axes[i][j].set_xticks([])
        if j != 0:
            axes[i][j].set_yticks([])
        # set white text color of ticks and color-bar
        axes[i][j].tick_params(colors=self.text_color, which='both')
        c_bar = axes[i][j].collections[0].colorbar
        c_bar.ax.tick_params(colors=self.text_color, which='both')

    def adjusted_color_bar(self, df_dictionary: Dict[int, pd.DataFrame], i: int) -> Tuple[int, int]:
        df_min, df_max = df_dictionary[self.boards[i]].min().min(), df_dictionary[self.boards[i]].max().max()
        threshold = 0.1
        if df_max - df_min < threshold:
            df_min = df_min - threshold if df_min - threshold > 0 else df_min
            df_max = df_max + threshold if df_max + threshold < int(self.zombies_per_episode) else df_max
        return df_min, df_max

    def name_conversion(self, name: str) -> str:
        converter = {'Const_vs_DDQN': self.agents_types[0],
                     'DoubleConst_vs_DDQN': self.agents_types[1],
                     'Uniform_vs_DDQN': self.agents_types[2],
                     'Gaussian_vs_DDQN': self.agents_types[3],
                     'DDQN_vs_Const': self.agents_types[0],
                     'DDQN_vs_DoubleConst': self.agents_types[1],
                     'DDQN_vs_Uniform': self.agents_types[2],
                     'DDQN_vs_Gaussian': self.agents_types[3]}
        if name in converter.keys():
            return converter[name]
        else:
            return name

    def calc_and_save_best_scores(self, df_dict: Dict[int, pd.DataFrame], best_results: Dict[int, pd.DataFrame], agent_type_index: int) \
            -> Dict[int, pd.DataFrame]:
        for board in list(df_dict.keys()):
            scores_df = df_dict[board]
            # find the maximum value and
            max_index_series = scores_df.where(lambda x: x == scores_df.max().max()).dropna(axis=1, how='all').dropna(axis=0, how='all').notnull().idxmax()
            t_update_m_size_tuple = list(max_index_series.to_dict().items())[0]
            best_results[board].loc[self.agents_types[agent_type_index], list(self.params.keys())[0]] = t_update_m_size_tuple[1]
            best_results[board].loc[self.agents_types[agent_type_index], list(self.params.keys())[1]] = t_update_m_size_tuple[0]
            best_results[board].loc[self.agents_types[agent_type_index], self.score_param] = scores_df.max().max()

        return best_results

    def add_board_annotation_and_tight(self, fig, axes):
        first_column_axes = axes[:] if len(axes.shape) == 1 else axes[:, 0]
        for ax, row in zip(first_column_axes, ['Board of {}'.format('\n' + str(board) + 'X' + str(board) + '  ') for board in self.boards]):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size=16, ha='right', va='center', color=self.text_color)
        plt.tight_layout()
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

    def save_results(self, axes, fig, player_type: str, best_results):
        self.add_board_annotation_and_tight(fig, axes)
        fig.savefig(os.path.join(self.docs_dir, 'DDQN_avg_test_rewards', 'DDQN as ' + player_type + '.png'), transparent=True)
        pd.concat(list(best_results.values())).round(2).to_csv(os.path.join(self.docs_dir, 'DDQN_avg_test_rewards', 'Best_of_' + player_type + '.csv'),
                                                               encoding='utf-8-sig')

    def plot_comparison_results(self, rewards_of_all_agents: Dict[str, Dict[str, Any]], best_results: Dict[int, pd.DataFrame], player_type: str):
        header = 'Comparison of ' + self.smart_agent + ' as ' + self.upper_first_letter(player_type) + ' Player vs. all Simple Agents'
        fig, axes = self.create_figure(len(self.boards), 1, 12, 9, header, 28)

        for i, board in enumerate(self.boards):
            best_result = best_results[board]
            for j, scenario in enumerate(list(rewards_of_all_agents.keys())):
                param_keys = list(self.params.keys())
                # the most generic way to validate the file we are looking for, matches all the parameters
                best_rewards_series_index = np.argwhere(np.logical_and(np.logical_and(
                    np.array(list(map(lambda x: x[param_keys[0]], rewards_of_all_agents[scenario]['info']))) == str(
                        self.params_types[0](best_result.loc[self.agents_types[j], param_keys[0]])),
                    np.array(list(map(lambda x: x[param_keys[1]], rewards_of_all_agents[scenario]['info']))) == str(
                        self.params_types[0](best_result.loc[self.agents_types[j], param_keys[1]]))),
                    np.array(list(map(lambda x: x['board'], rewards_of_all_agents[scenario]['info']))) == board))

                # summary the results in case of repetitions of the scenario
                number_of_repetitions = 0
                aggregated_best_rewards = np.zeros(len(rewards_of_all_agents[scenario][scenario][0]))
                for scenario_repetition in list(rewards_of_all_agents[scenario].keys()):
                    if scenario_repetition != 'info':
                        number_of_repetitions += 1
                        aggregated_best_rewards += rewards_of_all_agents[scenario][scenario_repetition][best_rewards_series_index[0][0]]
                if number_of_repetitions == 0:
                    print('Something is wrong! the scenario of: ' + scenario + ' is empty')

                average_best_rewards_along_episodes = aggregated_best_rewards / number_of_repetitions
                # plot moving average, set legend, set ticks to chosen color
                axes[i].plot(Utils.get_moving_average(self.moving_average_period, average_best_rewards_along_episodes), label=self.agents_types[j])
                axes[i].legend(loc='best')
                axes[i].tick_params(colors=self.text_color, which='both')
            # set edge_color of axis as white
            for spine in axes[i].spines.values():
                spine.set_edgecolor(self.text_color)
            if i == len(self.boards) - 1:
                # add x-label only in the last row
                axes[i].set_xlabel('Episode', color=self.text_color, fontsize=14)

        self.add_board_annotation_and_tight(fig, axes)
        fig.savefig(os.path.join(self.docs_dir, 'DDQN_avg_test_rewards', 'Comparison of DDQN as ' + player_type + '.png'), transparent=True)
