from experiments7_aics import objective_function, set_model_and_environment_parameters


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import torch
import numpy as np
import random

SEED = 1996
def set_seed(seed):
    if SEED is not None:
        # print(f'Seed set to {SEED}.')
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

environment = 'CartPole-v1'
environment = 'Acrobot-v1'

# metric = 'best_score'
metric = 'evaluation_score'

# 1. Read the CSV file into a dataframe
df_filtered = pd.read_csv("Experiments/Results/test_sept/experiments_log_filtered.csv")

# 2. Filter the "environment" column for "CartPole-v1"
# df_filtered = df_filtered[df_filtered["environment"] == environment]

# 3. Further filter the "model" column for "neuromodulated_hb"
# df_filtered = df_filtered[df_filtered["model"] == "static"]
df_filtered.loc[df_filtered["model"] == "abcd", "lambda_exp"] = -1
df_filtered.loc[df_filtered["model"] == "static", "lambda_exp"] = -2
df_filtered['testing'] = None

print(len(df_filtered))
for i, row in df_filtered.iterrows():
    set_seed(SEED)
    with open(row['filename'], 'rb') as f:
        x = pickle.load(f)
        best_solution = x['best_solution']
        print(x['population_size'])
        eval_tries = 1000
        model = row['model']
        env = row['environment']
        # lambda_value = row['lambda_decay']
        lambda_value = 0.01
        max_episode_steps, _, _ = set_model_and_environment_parameters(env, model)
        total_reward = objective_function(best_solution, tries = eval_tries, show=False, seed=SEED, model_name=model, environment_name=env, max_episode_steps=max_episode_steps, lambda_value=lambda_value)
        df_filtered.loc[i, 'testing'] = total_reward
        print(f'{i}: {env} - {model} - {total_reward}')

    # break
df_filtered.to_csv('Experiments/Results/test_sept/experiments_log_v4_1000.csv')




    
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
    df_filtered.groupby(["lambda_exp", "model"])[metric]  # replace with your metric
    .mean()
    .reset_index()
)

print("\nMean values of 'some_metric' by lambda_exp and model:")
print(mean_table)

# Plot with seaborn (colored by model)
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_filtered,
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