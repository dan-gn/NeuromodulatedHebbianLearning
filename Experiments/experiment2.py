"""
This is the experiment 2. 
I know this is a really bad name for an experiment but let me try to explain what is this experiment about.
Basically, I am testing the following:
    1. The artificial neural networks models, static and hebbian.
    2. The CMA-ES function which was currently made by my friend Chat-GPT.
    3. That I'm able to integrate my code with OpenAI Gymnasium environments.
And basically, that's it. 

"""


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

# Setting the parent path directory to call functions from other folders
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Models.static_neural_network import StaticNN
from Models.hebbian_learning import HebbianAbcdNN
from Optimisation.cma_es import CMA_ES
from Optimisation.cma_es_v2 import PureCMAES

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
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

# Environment options
ENV_NUMBER = 1
ENVIRONMENTS = []
ENVIRONMENTS.append('CartPole-v1')
ENVIRONMENTS.append('MountainCar-v0')
ENVIRONMENTS.append('LunarLander-v3')

# Optimisation parameters
POPULATION_SIZE = 50
ITERATIONS = 200
EVALUATIONS = POPULATION_SIZE * ITERATIONS
SIGMA = 0.5
TRIES = 10

# Evaluation parameters
SHOW_BEST = True    # Runs the best solution for EVAL_TRIES
EVAL_TRIES = 3

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MODEL AND ENVIRONMENT PARAMETERS
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Environment parameters
ENV = ENVIRONMENTS[ENV_NUMBER]
if ENV == 'CartPole-v1':
    MAX_EPISODE_STEPS = 1000
    STOP_CONDITION = -MAX_EPISODE_STEPS * TRIES
elif ENV == 'MountainCar-v0':
    MAX_EPISODE_STEPS = 200
    STOP_CONDITION = 110 * TRIES
elif ENV == 'LunarLander-v3':
    MAX_EPISODE_STEPS = 1000
    STOP_CONDITION = -200 * TRIES

# Neural Network parameters
MODEL = MODELS[MODEL_NUMBER]
if MODEL == 'static':
    HIDDEN_SIZES = [64, 32]
elif MODEL == 'abcd':
    HIDDEN_SIZES = [64]

# Compute action from environment
def compute_action(env_name, action):
    if env_name == 'CartPole-v1':
        action =  torch.sigmoid(action)
        return int(torch.round(action))
    elif env_name == 'MountainCar-v0':
        return int(nn.functional.hardtanh(action, 0, 2))
    elif env_name == 'LunarLander-v3':
        return int(nn.functional.hardtanh(action, 0, 2))

def get_model():
    if MODEL == 'static':
        model = StaticNN(input_size=env.observation_space.shape[0], output_size=1, hidden_sizes=HIDDEN_SIZES)
        n_variables = model.get_n_weights()
    elif MODEL == 'abcd':
        model =  HebbianAbcdNN(input_size=env.observation_space.shape[0], output_size=1, hidden_sizes=HIDDEN_SIZES, env_name=ENV)
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
    elif MODEL == 'abcd':
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

    for seed in range(1):

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
        # optimizer = CMA_ES(func=objective_function, dim=n_variables, sigma=SIGMA, popsize=POPULATION_SIZE)
        # optimizer = PureCMAES(objective_function, N=n_variables, sigma=SIGMA, stopfitness=STOP_CONDITION)
        optimizer = CMAES(pop_size=POPULATION_SIZE, sigma=SIGMA, restarts=EVALUATIONS, incpopsize=1, restart_from_best= 'True')
        termination = ValueBasedTermination(STOP_CONDITION, EVALUATIONS)
        problem = RLtask(n_var=n_variables, n_obj=1, xl=-1, xu=1, seed=seed)

        if READ_DATA:
            with open(f'Experiments/Results/{input_filename}', 'rb') as file:
                data = pickle.load(file)
            best_solution = data['best_solution']
            best_score = data['best_score']
        else:
            # Run optimization
            # best_solution, best_score = optimizer.optimize(iterations=ITERATIONS, stop_condition=STOP_CONDITION, seed=SEED, with_restarts=True)
            # best_solution, best_score = optimizer.optimize()
            res = minimize(
                problem,
                optimizer,
                # ('n_gen' , ITERATIONS),
                termination,
                seed = SEED,
                verbose = True
            )
            best_solution, best_score = res.X, res.F



        
        print("Best solution:", best_solution)
        print("Best score:", best_score)

        # Store experiment data
        if STORE_DATA and not READ_DATA:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f'Experiments/Results/exp2_march_pymoo/exp2_output_pymoo_{ENV}_{MODEL}_seed-{seed}_time-{timestamp}.pkl'
            output = {'best_solution': best_solution,
                    'best_score': best_score,
                    'hidden_size': HIDDEN_SIZES,
                    'population_size': POPULATION_SIZE,
                    'max_iterations': ITERATIONS,
                    # 'n_iterations' : optimizer.i,
                    'seed': SEED}
            with open(output_filename, 'wb') as file:
                pickle.dump(output, file)

        # Show best solution
        total_reward = objective_function(best_solution, tries = EVAL_TRIES, show=SHOW_BEST, seed=1996)
        print(f'Evaluation total reward {-total_reward}')

