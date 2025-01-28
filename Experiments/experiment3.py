import numpy as np
import torch 
import torch.nn as nn
import gymnasium as gym
import pickle
import datetime

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Models.static_neural_network import StaticNN
from Models.hebbian_learning import HebbianAbcdNN
from Utils.pymoo_utils import ValueBasedTermination

SEED = None
READ_DATA = False
STORE_DATA = False

# Model options
MODEL_NUMBER = 0
MODELS = []
MODELS.append('static')
MODELS.append('abcd')

MODEL = MODELS[MODEL_NUMBER]

# Environment options
ENV_NUMBER = 1
ENVIRONMENTS = []
ENVIRONMENTS.append('CartPole-v1')
ENVIRONMENTS.append('MountainCar-v0')
ENVIRONMENTS.append('LunarLander-v3')

# Neural Network parameters
HIDDEN_SIZES = [128, 64]
# HIDDEN_SIZES = [128]

# Optimisation parameters
MAX_EPISODE_STEPS = 1000
TRIES = 10
SIGMA = 0.5
POPULATION_SIZE = 50
ITERATIONS = 250
EVALUATIONS = POPULATION_SIZE * ITERATIONS

# Evaluation parameters
EVAL_TRIES = 3

# Environment parameters
ENV = ENVIRONMENTS[ENV_NUMBER]
if ENV == 'CartPole-v1':
    STOP_CONDITION = -MAX_EPISODE_STEPS * TRIES
elif ENV == 'MountainCar-v0':
    STOP_CONDITION = 115 * TRIES
elif ENV == 'LunarLander-v3':
    STOP_CONDITION = -200 * TRIES

# Compute action from environment
def compute_action(env_name, action):
    if env_name == 'CartPole-v1':
        action =  torch.sigmoid(action)
        return int(torch.round(action))
    elif env_name == 'MountainCar-v0':
        # action = torch.relu(action)
        # action = int(torch.round(action))
        # return min(2, action)
        return int(nn.functional.hardtanh(action, 0, 2))
    elif env_name == 'LunarLander-v3':
        # action = torch.relu(action)
        # action = int(torch.round(action))
        # return min(3, action)
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

# Objective function
def objective_function(x, tries = TRIES, show=False):
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
    action_count = np.zeros(env.action_space.n)
    with torch.no_grad():
        for i in range(tries):
            if SEED:
                observation, info = env.reset(seed=SEED)
            else:
                observation, info = env.reset()
            episode_over = False
            while not episode_over:
                action = model.forward(torch.Tensor(observation))
                action = compute_action(ENV, action)
                action_count[action] += 1
                observation, reward, terminated, truncated, info = env.step(action)
                episode_over = terminated or truncated
                result[i] += reward
    # print(action_count)
    env.close()
    return -np.sum(result)

from pymoo.core.problem import ElementwiseProblem
class RLtask(ElementwiseProblem):

    def __init__(self, n_var, n_obj, xl, xu):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = objective_function(x)

# Example usage
if __name__ == "__main__":

    if SEED:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)


    env = gym.make(ENV, render_mode="human", max_episode_steps=MAX_EPISODE_STEPS)
    model, n_variables = get_model()
    print(f'MODEL = {MODEL}, ENVIRONMENT = {ENV}')
    print(f'hidden = {HIDDEN_SIZES}, n_variables = {n_variables}, Layers = {len(model.layers)}')
    problem = RLtask(n_var=n_variables, n_obj=1, xl=-1, xu=1)

    # Instantiate the CMA-ES optimizer
    optimizer = CMAES(pop_size=POPULATION_SIZE, sigma=SIGMA, restarts=EVALUATIONS, incpopsize=1)
    from pymoo.termination import get_termination
    # termination = get_termination('n_eval', ITERATIONS)
    # termination = get_termination('f_min', STOP_CONDITION)
    termination = ValueBasedTermination(STOP_CONDITION, EVALUATIONS)


    if READ_DATA:
        input_filename = 'output_cartpole-v1_static_2025-01-21_15-00-44.pkl'
        with open(f'Results/{input_filename}', 'rb') as file:
            data = pickle.load(file)
        best_solution = data['best_solution']
        best_score = data['best_score']
    else:
        # Run optimization
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
        output_filename = f'Experiments/Results/output_pymoo_{ENV}_{MODEL}_{timestamp}.pkl'
        output = {'best_solution': best_solution,
                'best_score': best_score,
                'hidden_size': HIDDEN_SIZES,
                'population_size': POPULATION_SIZE,
                'iterations': ITERATIONS}
        with open(output_filename, 'wb') as file:
            pickle.dump(output, file)

    # Show best solution
    total_reward = objective_function(best_solution, tries = EVAL_TRIES, show=True)
    print(-total_reward)

