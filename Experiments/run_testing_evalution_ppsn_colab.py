from experiments10_ppsn_gcolab import objective_function, set_model_and_environment_parameters


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import torch
import numpy as np
import random

from concurrent.futures import ProcessPoolExecutor

SEED = 1996
eval_tries = 1000
CORES = 7

MODELS = []
MODELS.append('abcd')
MODELS.append('neuromodulated_hb')
MODELS.append('static')

ENVIRONMENTS = []
ENVIRONMENTS.append('MountainCar-v0')
ENVIRONMENTS.append('CartPole-v1')
ENVIRONMENTS.append('Acrobot-v1')

def set_seed(seed):
    if SEED is not None:
        # print(f'Seed set to {SEED}.')
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)


# Read the results dataframe
log_folder = f'../drive/MyDrive/PPSN26/Experiments/Results/test_ppsn_march/'
df = pd.concat([pd.read_csv(f'{log_folder}/experiments_log_colab_exp{colab_file}.csv') for colab_file in range(1, 4)], ignore_index=True)

new_file = log_folder + f'ppsn_testing_results_it_fix.csv'


# Filter by environment
# environment = ''
# df_filtered = df[df["environment"] == environment]

# Filter by model
# df_filtered = df_filtered[df_filtered["model"] == "static"]
# df_filtered.loc[df_filtered["model"] == "abcd", "lambda_exp"] = -1
# df_filtered.loc[df_filtered["model"] == "static", "lambda_exp"] = -2

df['testing'] = None

print(f'Total number of rows = {len(df)}')

for model in MODELS:
    print(f'Total number of rows of {model} = {(df['model'] == model).sum()}')
for env in ENVIRONMENTS:
    print(f'Total number of rows of {env} = {(df['environment'] == env).sum()}')

def run_single(i):
    set_seed(SEED)
    row = df.iloc[i]
    filename = log_folder + row['filename'].split('/')[-1]
    with open(filename, 'rb') as f:
        x = pickle.load(f)
        best_solution = x['best_solution']
        model = row['model']
        env = row['environment']
        lambda_value = row['lambda_decay']
        exp_seed = row['seed']
        max_episode_steps, _, _ = set_model_and_environment_parameters(env, model)
        # total_reward = objective_function(best_solution, tries = eval_tries, show=False, seed=SEED, model_name=model, environment_name=env, max_episode_steps=max_episode_steps, lambda_value=lambda_value)
        record = x['record']
        if env == 'Acrobot-v1':
            threshold = 750
            total_reward = len(record) - np.searchsorted(record[::-1], threshold, side='right')
        elif env == 'CartPole-v1':
            threshold = 1000
            total_reward = len(record) - np.searchsorted(record[::-1], threshold, side='right')
        elif env == 'MountainCar-v0':
            threshold = -5000
            total_reward = len(record) - np.searchsorted(record[::-1], threshold, side='right')
        else:
            total_reward = 0
        df.loc[i, 'testing'] = total_reward
        print(f'{i}: {env} - {model} -  {exp_seed} - {total_reward}')
    return total_reward


with ProcessPoolExecutor(max_workers=CORES) as executor:
    testing = list(executor.map(run_single, range(len(df))))

df['iteration_achieved_fix'] = testing

# for i, row in df.iterrows():
#     set_seed(SEED)
#     filename = log_folder + row['filename'].split('/')[-1]
#     # print(filename)
#     with open(filename, 'rb') as f:
#         x = pickle.load(f)
#         best_solution = x['best_solution']
#         # print(x['population_size'])
#         model = row['model']
#         env = row['environment']
#         lambda_value = row['lambda_decay']
#         exp_seed = row['seed']
#         # lambda_value = 0.01
#         max_episode_steps, _, _ = set_model_and_environment_parameters(env, model)
#         total_reward = objective_function(best_solution, tries = eval_tries, show=False, seed=SEED, model_name=model, environment_name=env, max_episode_steps=max_episode_steps, lambda_value=lambda_value)
#         df.loc[i, 'testing'] = total_reward
#         print(f'{i}: {env} - {model} -  {exp_seed} - {total_reward}')

    # break
df.to_csv(new_file)





    
metric = 'testing'


# 4. Make boxplots according to the "lambda_exp" column
# df_filtered.boxplot(column=metric, by=["lambda_exp"], figsize=(10, 6))  
# # Replace "some_metric" with the name of the column you want to plot values for

# plt.title("Boxplot of Some Metric by lambda_exp")
# plt.suptitle("")  # removes the automatic 'Boxplot grouped by lambda_exp' title
# plt.xlabel("lambda_exp")
# plt.ylabel("Testing fitness")
# plt.grid(False)
# plt.show()

# ---- Compute means ----
mean_table = (
    df.groupby(["lambda_exp", "model"])[metric]  # replace with your metric
    .mean()
    .reset_index()
)

print("\nMean values of 'some_metric' by lambda_exp and model:")
print(mean_table)

# Plot with seaborn (colored by model)
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="lambda_exp",
    y=metric,      # replace with your column to plot
    hue="model",          # different colors for models
    palette="Set2"        # you can try "Set1", "Dark2", "Pastel1", "tab10", etc.
)

plt.title("Boxplot of Some Metric by lambda_exp and model")
plt.xlabel("lambda_exp")
plt.ylabel("Score")
plt.legend(title="Model")
plt.grid(False)
plt.show()