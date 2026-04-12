""" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Import required libraries and functions
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

from experiments10_ppsn_gcolab import objective_function, set_model_and_environment_parameters

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import torch
import numpy as np
import random

from concurrent.futures import ProcessPoolExecutor


""" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ARGUMENTS
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

SEED = 1996
eval_tries = 1000
CORES = 47

MODELS = []
MODELS.append('abcd')
MODELS.append('neuromodulated_hb')
MODELS.append('static')
MODELS.append('static_double')

ENVIRONMENTS = []
ENVIRONMENTS.append('MountainCar-v0')
ENVIRONMENTS.append('CartPole-v1')
ENVIRONMENTS.append('Acrobot-v1')

# Read the results dataframe
log_folder = f'../drive/MyDrive/PPSN26/Experiments/Results/test_ppsn_march/'
# log_folder = f'Experiments/Results/test_ppsn_march/'

new_file = log_folder + f'ppsn_testing_results_final.csv'


""" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SECONDARY FUNCTIONS
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def set_seed(seed):
    if SEED is not None:
        # print(f'Seed set to {SEED}.')
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)


def filter_experiments(df):
    # Simple filters
    df = df[df["model"].isin(MODELS)]
    df = df[df["environment"].isin(ENVIRONMENTS)]
    df = df[df["cores"] == 47]
    df = df[df["population_size"] == 100]
    hidden_sizes = ['[64, 32]', '[150, 75]']
    df = df[df["hidden_size"].isin(hidden_sizes)]

    # Filter by number of iterations per each environment
    acrobot_parameters = {
        'environment' : 'Acrobot-v1',
        'max_iterations' : 100,
    }
    cartpole_parameters = {
        'environment' : 'CartPole-v1',
        'max_iterations' : 100,
    }
    mountaincar_parameters = {
        'environment' : 'MountainCar-v0',
        'max_iterations' : 500,
    }
    parameters = [acrobot_parameters, cartpole_parameters, mountaincar_parameters]
    filtered_dfs = [
        df[
            (df['environment'] == params['environment']) &
            (df['max_iterations'] == params['max_iterations'])
        ]
        for params in parameters
    ]
    df = pd.concat(filtered_dfs)

    parameters = [acrobot_parameters, cartpole_parameters, mountaincar_parameters]
    filtered_dfs = [
        df[
            (df['environment'] == params['environment']) &
            (df['max_iterations'] == params['max_iterations'])
        ]
        for params in parameters
    ]
    df = pd.concat(filtered_dfs)



    # Remove duplicates
    cols = ['model', 'environment', 'lambda_exp', 'seed']
    df = df[~df.duplicated(subset=cols, keep='last')]

    # Sort the dataframe
    df = df.sort_values(by=cols)

    return df

# Parallel run the experiments
def run_single(i, df):
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
        total_reward = objective_function(best_solution, tries = eval_tries, show=False, seed=SEED, model_name=model, environment_name=env, max_episode_steps=max_episode_steps, lambda_value=lambda_value)
        df.loc[i, 'testing'] = total_reward
        print(f'{i}: {env} - {model} -  {exp_seed} - {total_reward}')
    return total_reward


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MAIN FUNCTION
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

if __name__ == "__main__":

    # Read the CSV files
    df = pd.concat([pd.read_csv(f'{log_folder}/experiments_log_colab_exp{colab_file}.csv') for colab_file in range(1, 4)], ignore_index=True)

    # Choose which experiments to test
    df = filter_experiments(df)

    # Print the number of experiments to be sure everything is as expected
    print(f'Total number of rows = {len(df)}')
    for model in MODELS:
        print(f'Total number of rows of {model} = {(df['model'] == model).sum()}')
    for env in ENVIRONMENTS:
        print(f'Total number of rows of {env} = {(df['environment'] == env).sum()}')

    # Run the testings
    # with ProcessPoolExecutor(max_workers=CORES) as executor:
    #     testing = list(executor.map(run_single, range(len(df))))


    # from concurrent.futures import ProcessPoolExecutor
    from itertools import repeat

    with ProcessPoolExecutor(max_workers=CORES) as executor:
        testing = list(executor.map(run_single, range(len(df)), repeat(df)))

    # Store the new CSV
    df['testing'] = testing
    df.to_csv(new_file)


    # ORIGINAL VERSION (NOT PARALLEL)
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

