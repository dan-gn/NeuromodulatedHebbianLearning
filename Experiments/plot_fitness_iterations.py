import pandas as pd

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
                data.append([model, environment, content['best_score'], content['n_iterations'], content['record'], content['seed']])
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            pass

    columns = ['model', 'environment', 'best_score', 'n_iterations', 'record', 'seed']
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

    ax.plot(np.array(all_plots).mean(axis=0)[:it], label='Mean', linewidth=1.6)
    ax.axhline(THRESHOLD[env], color='red', linestyle='--', label='Objective')
    
    # ax.xlabel("Index")
    # ax.ylabel("Values")
    # ax.title(f"Plots of Column A for Category {model} - {env}")
    ax.legend()
    # plt.show()

if __name__ == "__main__":
    # Read CSV file
    file_path = "Experiments/gecco_experiments.csv"  
    df = read_csv_file(file_path)

    files = [filename.split('/')[-1] for filename in df['filename']]
    folder = 'Experiments/Results/gecco25_final'

    existing, missing = check_files_existence(folder, files)

    print('Existing files = ', len(existing))
    print('Missing files = ', len(missing))

    data = create_dataframe_from_pickles(folder, files)
    print(data.head())

    print(data.iloc[0]['record'])

    # Example plot usage
    model = MODELS[2]
    env = ENVIRONMENTS[0]
    model_names = {
        'static' : 'Fixed ANN',
        'abcd' : 'sHL', 
        'neuromodulated_hb' : 'tn-HL'
    }
    # plot_lists_for_category(data, model, env)


    # Generate sample data
    x = np.linspace(0, 10, 100)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))  # Create 3x3 subplots

    its = [40, 50, 250]

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
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness')
        if env == 'CarPole-v1':
            ax.set_ylim(-5000, -1000)
        elif env == 'MountainCar-v0':
            ax.set_ylim(1000, 4000)
        elif env == 'Acrobot-v1':
            ax.set_ylim(700, 1700)

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()





