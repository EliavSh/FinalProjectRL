from typing import List
import seaborn as sns

from postProcess.post_process import PostProcess

text_color = 'w'


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


if __name__ == "__main__":
    smart_agent = 'DDQN'
    learning_agent_topic = 'DdqnAgentInfo'
    boards = [10, 20, 30]
    params = {'memory_size': [3000, 4000, 5000], 'target_update': [500, 750, 1000]}
    post_process = PostProcess(smart_agent, learning_agent_topic, boards, params)

    df_dict, best_results = post_process.init_results_dicts()

    scenarios_list = post_process.get_list_dirs_relative_to_project_dir('final_results')
    for player_type in ['light', 'zombie']:
        ddqn_pos, dumb_pos = (0, 2) if player_type == 'light' else (2, 0)
        all_games_of_agent = split_list_by_words_in_specific_position(list_of_scenarios=scenarios_list, words=['DDQN'], position=ddqn_pos)
        unique_games = split_list_by_words_in_specific_position(all_games_of_agent[0], ['Const', 'DoubleConst', 'Uniform', 'Gaussian'], dumb_pos)

        fig, axes = post_process.create_figure(len(unique_games), player_type, 12, 9)

        rewards_of_all_agents = dict(zip(list(map(lambda x: x[0], unique_games)), list(map(lambda x: dict(zip([*x, 'info'], [[], [], []])), unique_games))))

        for j, two_players_game in enumerate(unique_games):
            mean_test_score_dict, rewards_of_all_agents = post_process.load_rewards(two_players_game, player_type, rewards_of_all_agents, j)
            # stream test mean values to df_dict
            for values in mean_test_score_dict.values():
                info = values[1]
                params_keys = list(params.keys())
                df_dict[info['board']].loc[int(info[params_keys[0]])][int(info[params_keys[1]])] = values[0]

            best_results = post_process.calc_and_save_best_scores(df_dict, best_results, j)

            # plot heat-map for every df
            for i in range(len(boards)):
                # adjust color-bar and plot heat-map
                value_min, value_max = post_process.adjusted_color_bar(df_dict, i)
                sns.heatmap(df_dict[boards[i]], ax=axes[i][j], annot=True, fmt="0.2f", vmin=value_min, vmax=value_max)
                post_process.set_subplots_title(axes, two_players_game, i, j)

        post_process.save_results(axes, fig, player_type, best_results)
        post_process.plot_comparison_results(rewards_of_all_agents, best_results, player_type)

    print('king')
