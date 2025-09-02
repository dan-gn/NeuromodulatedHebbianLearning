import pandas as pd
from tabulate import tabulate

from experiment5_gecco import objective_function, MODEL, ENV

MODELS = ['static', 'abcd', 'neuromodulated_hb']
ENVIRONMENTS = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
# environments = ['CartPole-v1', 'Acrobot-v1']

THRESHOLD  = {
    'CartPole-v1' : -5000,
    'MountainCar-v0' : 1100, 
    'Acrobot-v1' : 750
}

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print("CSV file read successfully!")
        # print(df.head())  # Display first few rows
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None
    
import os

def check_files_existence(folder_path, filenames):
    existing_files = []
    missing_files = []
    
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            existing_files.append(filename)
        else:
            missing_files.append(filename)
    
    return existing_files, missing_files

import pickle

def create_dataframe_from_pickles(folder_path, pickle_files):
    data = []
    
    for filename in pickle_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'rb') as f:
                content = pickle.load(f)
                model = [m for m in MODELS if m in file_path][0]
                environment = [e for e in ENVIRONMENTS if e in file_path][0]
                row = [model, environment, content['best_score'], content['best_solution'], content['n_iterations'], content['record'], content['seed']]
                # test = objective_function(content['best_solution'], tries = 100, show=False, seed=1996, model_name=model, environment_name=environment) 
                # print(f'{model} - {environment} - test = {test}')
                # row.append(test)
                data.append(row)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            pass

    # columns = ['model', 'environment', 'best_score', 'best_solution', 'n_iterations', 'record', 'seed', 'test']
    columns = ['model', 'environment', 'best_score', 'best_solution', 'n_iterations', 'record', 'seed']
    return pd.DataFrame(data, columns=columns)

import matplotlib.pyplot as plt
import numpy as np

def plot_lists_for_category(df, model, env, ax, it = 50):
    filtered_df = df[df['model'] == model]
    filtered_df = filtered_df[filtered_df['environment'] == env]
    all_plots = [] 
    # plt.figure(figsize=(10, 6))
    for index, row in filtered_df.iterrows():
        x = row['record']
        x = [xi if xi != 0 else row['best_score'] for xi in row['record']]
        all_plots.append(list(x))
        ax.plot(x[:it], color='gray', linewidth=0.8)

    ax.plot(np.array(all_plots).mean(axis=0)[:it], label='mean', linewidth=1.6)
    ax.axhline(THRESHOLD[env], color='red', linestyle='--', label='objective')
    
    # ax.xlabel("Index")
    # ax.ylabel("Values")
    # ax.title(f"Plots of Column A for Category {model} - {env}")
    ax.legend()
    # plt.show()

def get_table_row(df, model, env):
    filtered_df = df[df['model'] == model]
    filtered_df = filtered_df[filtered_df['environment'] == env]
    row = []
    row.append(model)
    row.append(env)
    row.append(filtered_df['best_score'].mean())
    row.append(filtered_df['best_score'].std())
    # total_reward = [objective_function(x, tries = 100, show=False, seed=1996, model_name=model, environment_name=env) for x in filtered_df['best_solution']] 
    # row.append(np.array(total_reward).mean())
    row.append(filtered_df['test'].mean())
    row.append(filtered_df['test'].std())
    row.append(filtered_df['n_iterations'].mean())
    row.append(filtered_df['n_iterations'].std())
    return row



import torch
import random

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42  # Use Type 42 (TrueType) fonts
mpl.rcParams['ps.fonttype'] = 42   # For saving as .ps if needed


if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # Read CSV file
    file_path = "Experiments/gecco_experiments.csv"  
    df = read_csv_file(file_path)

    files = [filename.split('/')[-1] for filename in df['filename']]
    folder = 'Experiments/Results/gecco25_final'

    existing, missing = check_files_existence(folder, files)

    print('Existing files = ', len(existing))
    print('Missing files = ', len(missing))

    data = create_dataframe_from_pickles(folder, files)
    # data.to_csv('final_results_gecco_april2.csv')
    # data = pd.read_csv('final_results_gecco_april2.csv')


    print(data.head())

    # print(data.iloc[0]['record'])

    # Example plot usage
    model = MODELS[2]
    env = ENVIRONMENTS[0]
    model_names = {
        'static' : 'Fixed ANN',
        'abcd' : 'sHL', 
        'neuromodulated_hb' : 'tm-HL'
    }
    # plot_lists_for_category(data, model, env)


    # Generate sample data
    x = np.linspace(0, 10, 100)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))  # Create 3x3 subplots
    fig.subplots_adjust(hspace=0.5)  # Adjust row spacing

    # import matplotlib.gridspec as gridspec

    # fig = plt.figure(figsize=(10, 10))
    # gs = gridspec.GridSpec(3, 3, height_ratios=[2, 2, 2], hspace=0.8)  # Increase hspace for more row separation

    # axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]


    # Titles for each row
    # fig.text(0.5, 0.88, "Row 1 Title", ha='center', fontsize=14, fontweight='bold')
    # fig.text(0.5, 0.58, "Row 2 Title", ha='center', fontsize=14, fontweight='bold')
    # fig.text(0.5, 0.28, "Row 3 Title", ha='center', fontsize=14, fontweight='bold')

    # Set row titles on the first column subplots and align them
    # axes[0, 0].set_title("Row 1 Title", fontsize=14, fontweight='bold', loc='left', pad=20)
    # axes[1, 0].set_title("Row 2 Title", fontsize=14, fontweight='bold', loc='left', pad=20)
    # axes[2, 0].set_title("Row 3 Title", fontsize=14, fontweight='bold', loc='left', pad=20)

    its = [40, 50, 250]

    table_data = []

    for i, ax in enumerate(axes.flat):
        # y = np.sin(x + i)  # Example function
        # ax.plot(x, y, label=f'Plot {i+1}')
        # ax.legend()
        # ax.set_title(f'Subplot {i+1}')
        # ax.grid(True)
        model = MODELS[int(i % 3)]
        env = ENVIRONMENTS[int(i / 3)]
        plot_lists_for_category(data, model, env, ax, it=its[int(i / 3)])
        # print(its[int(i % 3)])
        ax.set_title(f'{model_names[model]} - {env}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fitness values')
        if env == 'CarPole-v1':
            ax.set_ylim(-5000, -1000)
        elif env == 'MountainCar-v0':
            ax.set_ylim(1000, 4000)
        elif env == 'Acrobot-v1':
            ax.set_ylim(700, 1700)
        
        MODEL, ENV = model, env
        # row = get_table_row(data, model, env)
        # print(row)
        # table_data.append(row)

        if int(i % 3) == 0:
            # Add row title- using `fig.text`
            fig.text(0.5, 0.92 - i * 0.32, 'hola', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()  # Adjust layout for better spacing
    # plt.show()
    headers = ['Model', 'Environment', 'Training', 'std', 'Testing', 'std', 'Iterations', 'std']

    print(tabulate(table_data, headers=headers, tablefmt='grid'))


    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.5))
    fig.subplots_adjust(hspace=0.5)  # Adjust row spacing
    fig.subplots_adjust(wspace=0.45)  # Adjust row spacing
    fig.subplots_adjust(bottom=0.05)  # Adjust row spacing
    fig.subplots_adjust(top=0.925)  # Adjust row spacing
    fig.subplots_adjust(left=0.09)  # Adjust row spacing
    fig.subplots_adjust(right=0.985)  # Adjust row spacing

    # Sample data
    x = np.linspace(0, 10, 100)

    # Titles for rows
    # height = [0.92, 0.622, 0.332]
    height = [0.96, 0.629, 0.304]

    # Loop over rows and columns
    for i in range(3):  # Rows
        for j in range(3):  # Columns
            model = MODELS[j]
            env = ENVIRONMENTS[i]
            plot_lists_for_category(data, model, env, axes[i, j], it=its[i])
            # axes[i, j].plot(x, np.sin(x + i + j))  # Example plot
            # axes[i, j].set_title(f"Plot {i+1},{j+1}")  # Individual plot title
            axes[i, j].set_title(f'{model_names[model]}')
            axes[i, j].set_xlabel('Iterations')
            axes[i, j].set_ylabel('Fitness values')
            if env == 'CarPole-v1':
                axes[i, j].set_ylim(-5000, -1000)
            elif env == 'MountainCar-v0':
                axes[i, j].set_ylim(1000, 4000)
            elif env == 'Acrobot-v1':
                axes[i, j].set_ylim(700, 1700)

        # Add row title using `fig.text`
        # fig.text(0.5, 0.92 - i * 0.295, row_titles[i], ha='center', fontsize=14, fontweight='bold')
        fig.text(0.512, height[i], ENVIRONMENTS[i], ha='center', fontsize=12, fontweight='bold')

    plt.show()



