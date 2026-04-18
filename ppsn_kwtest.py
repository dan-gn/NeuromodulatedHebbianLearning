""" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Import required libraries and functions
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
from functools import reduce
from scipy.stats import kruskal
import scikit_posthocs as sp

""" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ARGUMENTS
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

MODELS = ['static', 'static_double', 'abcd', 'neuromodulated_hb']
ENVIRONMENTS = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0']

environment = ENVIRONMENTS[2]

# metric = 'best_score'
# metric = 'evaluation_score'
# metric = 'n_iterations'
metric = 'testing'
# metric = 'goal_achieved_it_fixed'
# metric = 'goal_achieved'

# TEST GECCO 25
# file_path = "final_results_gecco_april2.csv"
# TEST AICS25
# file_path = "Experiments/Results/test_aics/experiments_log_100tries_v1.csv"
# TEST PPSN26
# file_path = "Experiments/Results/test_ppsn_feb/ppsn_testing_results_1000tries.csv"  
# file_path = "Experiments/Results/test_ppsn_march/book5.csv"  
# file_path = "Experiments/Results/test_ppsn_march/ppsn_testing_results_march_mut.csv"  
file_path = "Experiments/Results/test_ppsn_april/ppsn_testing_results_final.csv"  

""" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SECONDARY FUNCTIONS
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def fix_df_names(df):
    # ---- Minor adjustments to the df ----
    df.loc[df["model"] == "abcd", "lambda_exp"] = -1
    df.loc[df["model"] == "static", "lambda_exp"] = -2
    df["model"] = df["model"].map({
        "neuromodulated_hb": "tm-HL",
        "static": "Fixed_ANN",
        "abcd": "sHL"
    })
    return df

def print_mean_table(df, metric, env, include_std = False):
    # ---- Compute means ----
    mean_table = (
        df.groupby(["lambda_exp", "model"])[metric]  # replace with your metric
        .mean()
        .reset_index()
    )
    print(f"\nMean values of {env} by lambda_exp and model:")
    print(mean_table)

    if include_std:
        # ---- Compute std ----
        std_table = (
            df.groupby(["lambda_exp", "model"])[metric]  # replace with your metric
            .std()
            .reset_index()
        )
        print(f"\nStd values of {env} by lambda_exp and model:")
        print(std_table)
    return mean_table, std_table

def kw_test(df, metric, env):
    df_env = df[df['environment'] == env]
    groups = [df_env.loc[df['lambda_exp'] == l][metric].to_numpy() for l in df['lambda_exp'].unique()]
    stat, p = kruskal(*groups)
    # print(f"Kruskal-Wallis H-statistic: {stat:.4f}")
    # print(f"p-value: {p:.4f}")
    return p < 0.05, p

def dunn_test(df, metric, env):
    df_env = df[df['environment'] == env]
    df_env = df[[metric, 'lambda_exp']]
    dunn_results = sp.posthoc_dunn(
        df_env,
        val_col=metric,
        group_col="lambda_exp",
        p_adjust="holm"  
    )
    dunn_results = dunn_results < 0.05
    dunn_overall = dunn_results.any().any()
    return dunn_overall, dunn_results



"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MAIN FUNCTION
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

if __name__ == "__main__":

    # Read the CSV file into a dataframe
    df= pd.read_csv(file_path)
    df = fix_df_names(df)

    '''
    Print and save the mean and std table
    '''
    # all_tables = []
    # for i, env in enumerate(ENVIRONMENTS):
    #     df_env = df[df["environment"] == env]
    #     # Print mean and std tables
    #     mean_table, std_table = print_mean_table(df_env, metric, env, include_std = True)
    #     folder_path = 'Experiments/Results/test_ppsn/'
    #     # Save std_table if computed
    #     if std_table is not None:
    #         # std_table.to_csv(folder_path + f"{env}_{metric}_std_table.csv", index=False)
    #         mean_table[f'{env}_std'] = std_table[metric]
    #     # Save mean_table
    #     # mean_table.to_csv(folder_path + f"{env}_{metric}_mean_table_v2.csv", index=False)
    #     mean_table = mean_table.rename(columns = {metric : f'{env}_mean'})
    #     all_tables.append(mean_table)
    # # full_table = pd.concat(all_tables, axis=1)
    # full_table = reduce(lambda left, right: pd.merge(left, right, on=['lambda_exp', 'model']), all_tables)
    # # full_table.to_csv(folder_path + f"{metric}_full_table.csv", index=False)

    print('')
    print(f'--- Metric = {metric} ---')
    print('')
    print('--- Kruskal Wallis H Test ---')
    results = {}
    for i, env in enumerate(ENVIRONMENTS):
        print(f'--- Environment = {env} ---')
        results[env], p = kw_test(df, metric, env)
        print(f'Result = {results[env]}')

    
    print('')
    print("--- Post-hoc test: Dunn's Test ---")
    posthoc = {}
    for i, env in enumerate(ENVIRONMENTS):
        if results[env] or True:
            print(f'--- Environment = {env} ---')
            posthoc[env], p = dunn_test(df, metric, env)
            print(f'Result = {posthoc[env]}')
            if posthoc[env]:
                print(p)
    
