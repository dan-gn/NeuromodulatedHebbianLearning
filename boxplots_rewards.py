""" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Import required libraries and functions
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from matplotlib.lines import Line2D


""" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ARGUMENTS
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

MODELS = ['static', 'abcd', 'neuromodulated_hb']
ENVIRONMENTS = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0']

environment = ENVIRONMENTS[2]

# metric = 'best_score'
# metric = 'evaluation_score'
# metric = 'n_iterations'
metric = 'testing'
# metric = 'goal_achieved_it'
# metric = 'goal_achieved_it_fix'
# metric = 'goal_achieved'
# metric = 'iteration_achieved_fix'
# metric = 'goal_achieved_fix'

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
    df.loc[df["model"] == "static", "lambda_exp"] = -3
    df.loc[df["model"] == "static_double", "lambda_exp"] = -2
    df["model"] = df["model"].map({
        "neuromodulated_hb": "tm-HL",
        "static": "Fixed-ANN",
        "static_double": "Fixed-ANN",
        "abcd": "sHL"
    })
    return df

def create_mean_table(df, metric, env, include_std = False, print_flag = True):

    # ---- Compute means ----
    mean_table = (
        df.groupby(["lambda_exp", "model"])[metric]  # replace with your metric
        .mean()
        .reset_index()
    )
    if print_flag:
        print(f"\nMean values of {env} by lambda_exp and model:")
        print(mean_table)

    if include_std:
        # ---- Compute std ----
        std_table = (
            df.groupby(["lambda_exp", "model"])[metric]  # replace with your metric
            .std()
            .reset_index()
        )
        if print_flag:
            print(f"\nStd values of {env} by lambda_exp and model:")
            print(std_table)
    return mean_table, std_table

def create_plot(df, metric):
    # Plot with seaborn (colored by model)
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="lambda_exp",
        y=metric,      # replace with your column to plot
        hue="model",          # different colors for models
        palette="Set2"        # you can try "Set1", "Dark2", "Pastel1", "tab10", etc.
    )

    plt.title(f'Total reward of each model on the testing set of seeds for {environment[:-3]}', fontsize=12)
    plt.xlabel(" ", fontsize=12)
    plt.ylabel("Total reward", fontsize=12)
    plt.legend(title="Model", fontsize=12)

    # Change tick labels
    labels = [r"$10^{-0}$", r"$10^{-0.5}$"]
    labels += [r"$10^{-1.0}$", r"$10^{-1.5}$"]
    labels += [r"$10^{-2.0}$", r"$10^{-2.5}$"]
    labels += [r"$10^{-3.0}$", r"$10^{-3.5}$"]
    labels += [r"$10^{-4.0}$"]
    labels = [fr'tm-HL $\lambda=${x}' for x in labels]
    labels = ['Fixed-ANN', 'sHL'] + labels

    plt.xticks(range(0, len(labels)), labels, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def create_plot_2(df, metric):
    # Plot with seaborn (colored by model)
    plt.figure(figsize=(10, 6))
    df["lambda_model"] = df["lambda_exp"].astype(str) + "_" + df["model"]
    sns.boxplot(
        data=df,
        x="lambda_model",
        y=metric,
        hue="environment",
        palette="Set2"
    )

    plt.title(f'Total reward of each model on the testing set of seeds for {environment[:-3]}', fontsize=12)
    plt.xlabel(" ", fontsize=12)
    plt.ylabel("Total reward", fontsize=12)
    plt.legend(title="Model", fontsize=12)

    # Change tick labels
    labels = [r"$10^{-0}$", r"$10^{-0.5}$"]
    labels += [r"$10^{-1.0}$", r"$10^{-1.5}$"]
    labels += [r"$10^{-2.0}$", r"$10^{-2.5}$"]
    labels += [r"$10^{-3.0}$", r"$10^{-3.5}$"]
    labels += [r"$10^{-4.0}$"]
    labels = [fr'tm-HL $\lambda=${x}' for x in labels]
    labels = ['Fixed-ANN', 'sHL'] + labels

    plt.xticks(range(0, len(labels)), labels, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def create_plot_3(df, metric):
    # Choose color palette
    # colors = sns.color_palette()
    colors = [
    '#4E79A7', '#F28E2B', '#E15759',
    '#76B7B2', '#59A14F', '#EDC948',
    '#B07AA1', '#FF9DA7', '#9C755F'
    ]



    new_palette = [colors[4], colors[2], colors[0]]
    target_color = colors[1]
    # Boxplots
    g = sns.catplot(
        data=df,
        x="lambda_exp",
        y=metric,
        hue="model",
        row="environment",   # one plot per environment
        kind="box",
        palette=new_palette,
        sharey=False,
        row_order=ENVIRONMENTS,
        aspect = 1,
        # legend=None
    )

    # Figure Size
    g.fig.set_size_inches(8, 9)
    
    if metric in ['best_score', 'testing']:
        # Target value line
        tries = 10 if metric == 'best_score' else 1000
        target_value = {'Acrobot-v1': 75, 'CartPole-v1': -500, 'MountainCar-v0': 110}
        for i, ax in enumerate(g.axes.flat):
            # ax.axhline(y=target_value[ENVIRONMENTS[i]] * tries, color=colors[1], linestyle='--', linewidth=1, label='Target' if i == 0 else None, zorder=0)
            ax.axhline(y=target_value[ENVIRONMENTS[i]] * tries, color=target_color, linestyle='--', linewidth=1, label='Target' if i == 0 else None)
        # g._legend = g.axes.flat[0].legend()

    # Subplot titles
    if metric == 'best_score':
        g.fig.suptitle(f'Total reward of each model and testing environment during training', fontsize=14)
        g.set_ylabels("Total reward", fontsize=10)
    elif metric == 'testing':
        g.fig.suptitle(f'Total reward of each model and testing environment over the testing set of seed', fontsize=14)
        g.set_ylabels("Total reward", fontsize=10)
        ylims = {
            "Acrobot-v1": (70000, 105000),
            "CartPole-v1": (-550000, -50000),
            "MountainCar-v0": (100000, 400000)
        }
        for ax, env in zip(g.axes.flat, ENVIRONMENTS):
            ax.set_ylim(ylims[env])
    elif metric in ['goal_achieved_it_fix', 'iteration_achieved_fix']:
        g.fig.suptitle(f'Number of iterations to achieved the testing environment goal for each model', fontsize=14)
        g.set_ylabels("Number of iterations", fontsize=10)
        ylims = {
            "Acrobot-v1": (0, 120),
            "CartPole-v1": (0, 80),
            "MountainCar-v0": (0, 550)
        }
        for ax, env in zip(g.axes.flat, ENVIRONMENTS):
            ax.set_ylim(ylims[env])
    g.set_titles("{row_name}", fontsize=12)

    # Labels 
    g.set_yticklabels(fontsize=10)
    g.set_xlabels(" ", fontsize=12)

    # Legend
    old_legend = g._legend
    handles = old_legend.legend_handles
    labels = [t.get_text() for t in old_legend.texts]
    line_handle = Line2D([0], [0], color=target_color, linestyle='--', linewidth=1)
    handles.append(line_handle)
    if metric in ['best_score', 'testing']:
        labels.append("Target reward")
    old_legend.remove()
    g.fig.legend(
        handles,
        labels,
        # title="Model",
        loc="center right",
        bbox_to_anchor=(1.0, 0.85),
        fontsize=10,
        title_fontsize=10,
        frameon=False
    )
    # g.add_legend()
    # g._legend.set_frame_on(True)
    # g._legend.set_title("Model")
    # g._legend.get_title().set_fontsize(10)
    # g._legend.set_bbox_to_anchor((0.99, 0.85))
    # plt.legend(title="Model", fontsize=10, bbox_to_anchor=(1.0, 0.85))

    # Change tick labels
    labels = [r"$1$", r"$10^{-0.5}$"]
    labels += [r"$10^{-1.0}$", r"$10^{-1.5}$"]
    labels += [r"$10^{-2.0}$", r"$10^{-2.5}$"]
    labels += [r"$10^{-3.0}$", r"$10^{-3.5}$"]
    labels += [r"$10^{-4.0}$"]
    labels = [fr'tm-HL $\lambda=${x}' for x in labels]
    labels = ['Fixed-ANN 1', 'Fixed-ANN 2', 'sHL'] + labels
    plt.xticks(range(0, len(labels)), labels, rotation=45, fontsize=10)

    # Design
    plt.grid(False)
    plt.tight_layout()
    g.fig.subplots_adjust(
        top=0.92,    # space for the figure title
        bottom=0.12, # space at the bottom
        left=0.12,   # space on the left
        right=0.78, # leave space on the right for the legend
        # hspace=0.3, # vertical space between rows
        # wspace=0.2  # horizontal space between columns
    )
    plt.show()

def create_plot_4(df, metric):
    # colors = sns.color_palette()
    colors = [
    '#4E79A7', '#F28E2B', '#E15759',
    '#76B7B2', '#59A14F', '#EDC948',
    '#B07AA1', '#FF9DA7', '#9C755F'
    ]
    new_palette = [colors[0], colors[2], colors[4]]


    df[metric] = df[metric] * 30
    plt.figure(figsize=(8, 6))
    g = sns.barplot(
        data = df,
        x = "lambda_exp",
        y = metric,
        hue = "environment",
        hue_order = ENVIRONMENTS,
        errorbar = None,
        palette = new_palette
    )

    # g._legend.set_frame_on(False)
    plt.title(f'Success rate during optimisation process', fontsize=14)
    plt.xlabel(" ", fontsize=2)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title="Environment", fontsize=12, bbox_to_anchor=(1.02, 1), frameon=False)

    # Change tick labels
    labels = [r"$1$", r"$10^{-0.5}$"]
    labels += [r"$10^{-1.0}$", r"$10^{-1.5}$"]
    labels += [r"$10^{-2.0}$", r"$10^{-2.5}$"]
    labels += [r"$10^{-3.0}$", r"$10^{-3.5}$"]
    labels += [r"$10^{-4.0}$"]
    labels = [fr'tm-HL $\lambda=${x}' for x in labels]
    labels = ['Fixed-ANN 1', 'Fixed-ANN 2', 'sHL'] + labels

    plt.xticks(range(0, len(labels)), labels, rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MAIN FUNCTION
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

if __name__ == "__main__":

    # Read the CSV file into a dataframe
    df= pd.read_csv(file_path)
    df = fix_df_names(df)

    # Filter the "environment" or "model"
    # df_filtered = df[df["environment"] == environment]
    # df_filtered = df_filtered[df_filtered["model"] == "neuromodulated_hb"]

    '''
    Print and save the mean and std table
    '''
    all_tables = []
    for i, env in enumerate(ENVIRONMENTS):
        df_env = df[df["environment"] == env]
        # Print mean and std tables
        mean_table, std_table = create_mean_table(df_env, metric, env, include_std = True, print_flag = False)
        folder_path = 'Experiments/Results/test_ppsn_april/'
        if std_table is not None:
            # Save std_table if computed
            # std_table.to_csv(folder_path + f"{env}_{metric}_std_table.csv", index=False)
            mean_table[f'{env}_std'] = std_table[metric]
        # Save mean_table
        # mean_table.to_csv(folder_path + f"{env}_{metric}_mean_table_v2.csv", index=False)
        mean_table = mean_table.rename(columns = {metric : f'{env}_mean'})
        all_tables.append(mean_table)
    full_table = reduce(lambda left, right: pd.merge(left, right, on=['lambda_exp', 'model']), all_tables)
    
    save_table_flag = False
    if save_table_flag:
        full_table.to_csv(folder_path + f"{metric}_full_table.csv", index=False)

    '''
    Show the plots
    '''
    create_plots_flag = True
    if create_plots_flag:
        df_filtered = df

        # Crete plot
        if metric == "goal_achieved":
            create_plot_4(df_filtered, metric)
        else:
            create_plot_3(df_filtered, metric)

