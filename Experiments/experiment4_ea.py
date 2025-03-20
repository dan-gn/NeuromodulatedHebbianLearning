
""" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Import required libraries and functions
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Libraries
import numpy as np
import torch 
import torch.nn as nn
import gymnasium as gym
import pickle
import datetime
import random
import time

# Setting the parent path directory to call functions from other folders
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Models.static_neural_network import StaticNN
from Models.hebbian_learning import HebbianAbcdNN
from Models.neuromodulated_hb import TimeBasedNeuromodulatedHebbianNN
# from Optimisation.cma_es import CMA_ES
# from Optimisation.cma_es_v2 import PureCMAES
from Optimisation.ea import EvolutionaryAlgorithm

from experiments_log import append_line_to_csv

# from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from Utils.pymoo_utils import ValueBasedTermination
from pymoo.core.problem import ElementwiseProblem

class RLtask(ElementwiseProblem):

    def __init__(self, n_var, n_obj, xl, xu, seed):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.seed = seed

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = objective_function(x, seed=self.seed)


""" 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ARGUMENTS
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Experiment parameters
SEED = None
READ_DATA = False   # Read data from input_filename
STORE_DATA = True   # Store data on output_filename

input_filename = 'output_MountainCar-v0_static_2025-01-27_21-10-50.pkl'

# Model options
MODEL_NUMBER = 0
MODELS = []
MODELS.append('static')
MODELS.append('abcd')
MODELS.append('neuromodulated_hb')

# Environment options
ENV_NUMBER = 3
ENVIRONMENTS = []
ENVIRONMENTS.append('LunarLander-v3')
ENVIRONMENTS.append('CartPole-v1')
ENVIRONMENTS.append('MountainCar-v0')
ENVIRONMENTS.append('Acrobot-v1')

# Optimisation parameters
POPULATION_SIZE = 100
ITERATIONS = 1000
EVALUATIONS = POPULATION_SIZE * ITERATIONS
TRIES = 20
MAX_STAGNMENT = 50
LAMBDA_DECAY = 0.01

# Evaluation parameters
SHOW_BEST = True    # Runs the best solution for EVAL_TRIES
EVAL_TRIES = TRIES

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MODEL AND ENVIRONMENT PARAMETERS
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Environment parameters
ENV = ENVIRONMENTS[ENV_NUMBER]
MODEL = MODELS[MODEL_NUMBER]
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
    if model == 'static':
        HIDDEN_SIZES = [64, 32]
    elif model == 'abcd' or model == 'neuromodulated_hb':
        HIDDEN_SIZES = [64, 32]
    return MAX_EPISODE_STEPS, STOP_CONDITION, HIDDEN_SIZES

MAX_EPISODE_STEPS, STOP_CONDITION, HIDDEN_SIZES = set_model_and_environment_parameters(ENV, MODEL)

# Compute action from environment
def compute_action(env_name, action):
    if env_name == 'CartPole-v1':
        action =  torch.sigmoid(action)
        return int(torch.round(action))
    elif env_name in ['MountainCar-v0', 'Acrobot-v1']:
        return int(nn.functional.hardtanh(action, 0, 2))
    elif env_name == 'LunarLander-v3':
        return int(nn.functional.hardtanh(action, 0, 3))

# Get the model and the number of variables
def get_model():
    if MODEL == 'static':
        model = StaticNN(input_size=env.observation_space.shape[0], output_size=1, hidden_sizes=HIDDEN_SIZES)
        n_variables = model.get_n_weights()
    elif MODEL == 'abcd':
        model =  HebbianAbcdNN(input_size=env.observation_space.shape[0], output_size=1, hidden_sizes=HIDDEN_SIZES, env_name=ENV)
        n_variables = model.get_n_weights() * 5
    elif MODEL == 'neuromodulated_hb':
        model =  TimeBasedNeuromodulatedHebbianNN(input_size=env.observation_space.shape[0], output_size=1, hidden_sizes=HIDDEN_SIZES, env_name=ENV, lambda_decay=0.001)
        n_variables = model.get_n_weights() * 5
    else:
        raise ValueError('Model not found.')
    return model, n_variables

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
OBJECTIVE FUNCTION
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def objective_function(x, tries = TRIES, show=False, seed = None):
    if show:
        env = gym.make(ENV, render_mode="human", max_episode_steps=MAX_EPISODE_STEPS)
    else:
        env = gym.make(ENV, max_episode_steps=MAX_EPISODE_STEPS)
    model, n_variables = get_model()
    if MODEL == 'static':
        model.update_weights(torch.Tensor(x))
    elif MODEL == 'abcd' or MODEL == 'neuromodulated_hb':
        model.update_weights(torch.rand(n_variables))
        model.update_hebbian(torch.Tensor(x))
    result = np.zeros(tries)
    with torch.no_grad():
        for i in range(tries):
            if seed is not None:
                observation, info = env.reset(seed=seed+i)
            else:
                observation, info = env.reset()
            episode_over = False
            while not episode_over:
                action = model.forward(torch.Tensor(observation))
                action = compute_action(ENV, action)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_over = terminated or truncated
                result[i] += reward
            if ENV_NUMBER == 1 and truncated == True:
                result[i] += -200
    env.close()
    return -np.sum(result)


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MAIN FUNCTION
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":

    for ENV in ENVIRONMENTS:
        for MODEL in MODELS:
            MAX_EPISODE_STEPS, STOP_CONDITION, HIDDEN_SIZES = set_model_and_environment_parameters(ENV, MODEL)
            for seed in range(30):
                start_time = time.time()

                SEED = seed

                if SEED is not None:
                    print(f'Seed set to {SEED}.')
                    torch.manual_seed(SEED)
                    torch.cuda.manual_seed(SEED)
                    np.random.seed(SEED)
                    random.seed(SEED)

                env = gym.make(ENV, render_mode="human", max_episode_steps=MAX_EPISODE_STEPS)
                model, n_variables = get_model()
                print(f'MODEL = {MODEL}, ENVIRONMENT = {ENV}')
                print(f'hidden = {HIDDEN_SIZES}, n_variables = {n_variables}, Layers = {len(model.layers)}')

                # Instantiate the CMA-ES optimizer
                optimizer = EvolutionaryAlgorithm(n_variables=n_variables, objective_function=objective_function, max_iterations=ITERATIONS, population_size=POPULATION_SIZE, max_stagnment=MAX_STAGNMENT)

                if READ_DATA:
                    with open(f'Experiments/Results/{input_filename}', 'rb') as file:
                        data = pickle.load(file)
                    best_solution = data['best_solution']
                    best_score = data['best_score']
                else:
                    # Run optimization
                    best_solution, best_score = optimizer.run(STOP_CONDITION, SEED)

                
                print("Best solution:", best_solution)
                print("Best score:", best_score)

                # Show best solution
                total_reward = objective_function(best_solution, tries = EVAL_TRIES, show=SHOW_BEST, seed=1996)
                print(f'Evaluation total reward {total_reward}')

                # Store experiment data
                if STORE_DATA and not READ_DATA:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    output_filename = f'Experiments/Results/exp4_march_ea/exp4_output_ea_{ENV}_{MODEL}_seed-{seed}_time-{timestamp}.pkl'
                    output = {'best_solution': best_solution,
                            'best_score': best_score,
                            'hidden_size': HIDDEN_SIZES,
                            'population_size': POPULATION_SIZE,
                            'max_iterations': ITERATIONS,
                            'n_iterations' : optimizer.i,
                            'record' : optimizer.record, 
                            'seed': SEED}
                    with open(output_filename, 'wb') as file:
                        pickle.dump(output, file)

                    log_file = f'Experiments/experiments_log.csv'
                    new_line = {
                        'filename' : output_filename,
                        'algorithm' : 'EA',
                        'environment' : ENV,
                        'model' : MODEL,
                        'seed' : SEED,
                        'population_size': POPULATION_SIZE,
                        'max_iterations': ITERATIONS,
                        'n_iterations' : optimizer.i,
                        'best_score' : best_score, 
                        'hidden_size' : str(HIDDEN_SIZES),
                        'max_stagnment' : MAX_STAGNMENT,
                        'lambda_decay' : LAMBDA_DECAY,
                        'tries' : TRIES,
                        'evaluation_score' : total_reward,
                        'eval_tries' : EVAL_TRIES,
                        'time' : time.time() - start_time
                    }
                    append_line_to_csv(log_file, new_line)

