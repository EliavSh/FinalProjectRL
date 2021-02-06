# get a list of final_results' directories

# split a list directories to types, based on: "starts with ..." (DDQN / not DDQN)

# split a list of directories to types, based on: "ends with ..." (Uniform / Gaussian / Const / DoubleConst) (watchout the _1 ending)

# [[[DDQN vs Uniform , DDQN vs Uniform_1], [DDQN vs Gaussian, DDQN vs Gaussian_1]...], [[Uniform vs DDQN, Uniform_1 vs DDQN], [Gaussian vs DDQN, Gaussian_1 vs DDQN]...]]

# load rewards:
## for each inner list:
## init df of size (1000,27)
### for each value:
### validations: config has all values: board, target update, memory
#### load the df by mean: each column named after concatenation of board size, target update and memory size (from config)
# --#--#--# the output should be a flat list with a dict for each list with key of the name of the scenario and value of the df:
# --#--#--# ["ddqn_vs_uniform" : df, "ddqn_vs_gaussian" : df, ..., "uniform_vs_ddqn" : df, "gaussian_vs_ddqn" : df, ...]

# get_mean_test_score:
## for each inner dict:
### map the value to a list of dfs (as the number of game boards) in such way:
### each df consists of target_update (columns) and memory_size (rows) with value of mean_test_reward (3 by 3 in our case)

# print_tables
## for each list of dfs, plot tables in concatenated way (9 by 3 in our case. with separations of different board sizes)
