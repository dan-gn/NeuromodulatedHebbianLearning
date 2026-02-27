""" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Import required libraries and functions
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Libraries
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import datetime

import time
from concurrent.futures import ProcessPoolExecutor


""" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ARGUMENTS
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# Experiment parameters
SEED = 0
MODEL = 'static'

TRIES = 1
SHOW_BEST = False    # Store recorded file

LAMBDA = 0

# Environment options
ENV_NUMBER = 1
ENVIRONMENTS = []
ENVIRONMENTS.append('MountainCar-v0')
ENVIRONMENTS.append('LunarLander-v3')
ENVIRONMENTS.append('CartPole-v1')
ENVIRONMENTS.append('Acrobot-v1')
ENV = ENVIRONMENTS[ENV_NUMBER]

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Static Artificial Neural Network model
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

class StaticNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes = [128, 64]) -> None:
        super(StaticNN, self).__init__()
        self.layers = []
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(self.layer_sizes)-1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
        self.activation = nn.Tanh()
        self.ABCD = torch.Tensor(input_size, output_size, 5)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        return self.layers[-1](x)
    
    def get_layer_n_weights(self, layer):
        return layer.weight.shape[0] * layer.weight.shape[1]

    def get_n_weights(self):
        n_weights = 0
        for layer in self.layers:
            n_weights += self.get_layer_n_weights(layer)
        return n_weights

    def update_weights(self, weights):
        a = 0
        for i, layer in enumerate(self.layers):
            b = a + self.get_layer_n_weights(layer)
            w = torch.reshape(weights[a:b], (layer.weight.shape))
            self.layers[i].weight = nn.Parameter(w)
            a = b

    def print_weights(self):
        for i, layer in enumerate(self.layers):
            print(f'Layer {i}')
            print(layer.weight)


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MODEL AND ENVIRONMENT PARAMETERS
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# Environment parameters
def set_model_and_environment_parameters(env, model):
    if env == 'CartPole-v1':
        MAX_EPISODE_STEPS = 500
        STOP_CONDITION = -MAX_EPISODE_STEPS * TRIES
    elif env == 'MountainCar-v0':
        MAX_EPISODE_STEPS = 200
        STOP_CONDITION = 110 * TRIES
    elif env == 'LunarLander-v3':
        MAX_EPISODE_STEPS = 500
        STOP_CONDITION = -200 * TRIES
    elif env == 'Acrobot-v1':
        MAX_EPISODE_STEPS = 500
        STOP_CONDITION = 100 * TRIES
    # Neural Network parameters
    HIDDEN_SIZES = [64, 32]
    return MAX_EPISODE_STEPS, STOP_CONDITION, HIDDEN_SIZES

# Compute action from environment
def compute_action(env_name, action):
    if env_name == 'CartPole-v1':
        action =  torch.sigmoid(action)
        return int(torch.round(action))
    elif env_name in ['MountainCar-v0', 'Acrobot-v1']:
        probs = F.softmax(action, dim=0)  
        return int(probs.argmax())
    elif env_name == 'LunarLander-v3':
        probs = F.softmax(action, dim=0)  
        return int(probs.argmax())

# Get the model and the number of variables
def get_model(output_size, model_name, env, env_name, lambda_value, hidden_sizes):
    if model_name == 'static':
        model = StaticNN(input_size=env.observation_space.shape[0], output_size=output_size, hidden_sizes=hidden_sizes)
        n_variables = model.get_n_weights()
    else:
        raise ValueError('Model not found.')
    return model, n_variables

def get_output_size(env):
    if env == 'CartPole-v1':
        return 1
    elif env in ['MountainCar-v0', 'Acrobot-v1']:
        return 3
    elif env == 'LunarLander-v3':
        return 4


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
OBJECTIVE FUNCTION
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def objective_function(x, model_name = MODEL, environment_name = ENV, tries = TRIES, show=False, seed = None, max_episode_steps = None, lambda_value=None):
    max_episode_steps, _, hidden_sizes = set_model_and_environment_parameters(environment_name, model_name)
    if show:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_name = f'{environment_name}_{model_name}_starting_seed{seed}_date{timestamp}'
        tmp_env = gym.make(environment_name, render_mode="rgb_array", max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordVideo(env=tmp_env, name_prefix=video_name, video_folder='SmallTest/')

    else:
        env = gym.make(environment_name, max_episode_steps=max_episode_steps)
    output_size = get_output_size(environment_name)
    model, n_variables = get_model(output_size, model_name, env, environment_name, lambda_value, hidden_sizes=hidden_sizes)
    model.update_weights(torch.Tensor(x))
    result = np.zeros(tries)
    action_record = []
    with torch.no_grad():
        for i in range(tries):
            if seed is not None:
                observation, info = env.reset(seed=seed+i)
            else:
                observation, info = env.reset()
            episode_over = False
            # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # video_name = f'{environment_name}_{model_name}_seed{seed+i}_date{timestamp}'
            # env.start_recording(video_name=video_name)
            while not episode_over:
                action = model.forward(torch.Tensor(observation))
                action = compute_action(environment_name, action)
                action_record.append(action)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_over = terminated or truncated
                result[i] += reward
            if environment_name == 'MountainCar-v0' and truncated == True:
                result[i] += -200
    # env.stop_recording()
    env.close()
    # print(np.array(action_record).mean())
    return -np.sum(result)


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MAIN FUNCTION
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""


import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor

def run_parallel_rewards(n):
    # 1. 'Freeze' the arguments that don't change
    # This creates a new version of the function that only needs the 'seed'
    partial_obj_func = partial(
        objective_function, 
        random_solution, 
        tries=TRIES, 
        show=SHOW_BEST, 
        model_name=MODEL, 
        environment_name=ENV
    )

    # 2. Generate the dynamic part (the seeds)
    seeds = [SEED + i for i in range(n)]

    # 3. Execute in parallel
    with ProcessPoolExecutor() as executor:
        # executor.map replaces your 'for' loop and 'append'
        rewards = list(executor.map(partial_obj_func, seeds))

    total_reward = np.sum(rewards)
    return total_reward

def task(n):
    return n ** 2

def run_single(i, random_solution, TRIES, SHOW_BEST, SEED, MODEL, ENV):
    return objective_function(
        random_solution,
        tries=TRIES,
        show=SHOW_BEST,
        seed=SEED + i,
        model_name=MODEL,
        environment_name=ENV
    )

if __name__ == "__main__":

    # Set random seed for all libraries
    if SEED is not None:
        # print(f'Seed set to {SEED}.')
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)

    # Get model and environment parameters
    MAX_EPISODE_STEPS, STOP_CONDITION, HIDDEN_SIZES = set_model_and_environment_parameters(ENV, MODEL)
    env = gym.make(ENV, render_mode="human", max_episode_steps=MAX_EPISODE_STEPS)
    output_size = get_output_size(ENV)
    model, n_variables = get_model(output_size, MODEL, env, ENV, LAMBDA, hidden_sizes=HIDDEN_SIZES)
    # print(f'MODEL = {MODEL}, ENVIRONMENT = {ENV}, LAMBDA = {LAMBDA}')
    # print(f'hidden = {HIDDEN_SIZES}, n_variables = {n_variables}, Layers = {len(model.layers)}, Stopping criteria = {STOP_CONDITION}')

    # Get a random array as for the ANN weights 
    initial_value_range = 5 
    random_solution = np.random.uniform(-initial_value_range, initial_value_range, n_variables)

    # Run the experiment
    # total_reward = objective_function(random_solution, tries = TRIES, show=SHOW_BEST, seed=SEED, model_name=MODEL, environment_name=ENV)

    n = 1000

    start_time = time.time()
    rewards = []
    for i in range(n):
        reward = objective_function(random_solution, tries = TRIES, show=SHOW_BEST, seed=SEED + i, model_name=MODEL, environment_name=ENV)
        rewards.append(reward)
    total_reward = np.sum(rewards)
    end_time = time.time() - start_time

    # print(f'Evaluation total reward {total_reward}')
    print(f'Evaluation total time = {end_time}')



    # # Always wrap the execution in this guard to prevent recursive process spawning
    # start_time = time.time()
    # total_reward = run_parallel_rewards(n)
    # end_time = time.time() - start_time
    # print(f'Evaluation total reward {total_reward}')
    # print(f'Evaluation total time {end_time}')

    # # nums = [random_solution] * n
    # nums = [i for i in range(n)]

    # start_time = time.time()
    # with ProcessPoolExecutor() as executor:
    #     results = list(executor.map(lambda x: objective_function(random_solution, tries=TRIES, show=SHOW_BEST, seed=SEED+x, model_name=MODEL, environment_name=ENV), nums))
    # end_time = time.time() - start_time
    # print(f'Evaluation total time {end_time}')

    for w in range(1, 21):
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=w) as executor:
            rewards = list(
                executor.map(
                    partial(
                        run_single,
                        random_solution=random_solution,
                        TRIES=TRIES,
                        SHOW_BEST=SHOW_BEST,
                        SEED=SEED,
                        MODEL=MODEL,
                        ENV=ENV,
                    ),
                    range(n)
                )
            )
        total_reward = np.sum(rewards)
        end_time = time.time() - start_time
        print(f'Workers = {w} - Evaluation total time = {end_time}')
