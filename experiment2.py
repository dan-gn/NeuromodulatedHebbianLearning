import numpy as np
import torch 
import gymnasium as gym

from Algorithms.hebbian_learning import StaticNN, HebbianAbcdNN
from Algorithms.cma_es import CMA_ES

SEED = None

# Model options
MODEL_NUMBER = 1
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

# Optimisation parameters
MAX_EPISODE_STEPS = 1000
TRIES = 3
SIGMA = 0.5
POPULATION_SIZE = 50
ITERATIONS = 100

# Evaluation parameters
EVAL_TRIES = 3

# Environment parameters
ENV = ENVIRONMENTS[ENV_NUMBER]
if ENV == 'CartPole-v1':
    STOP_CONDITION = -MAX_EPISODE_STEPS * TRIES
elif ENV == 'MountainCar-v0':
    STOP_CONDITION = 50 * TRIES
elif ENV == 'LunarLander-v3':
    STOP_CONDITION = -200 * TRIES

# Compute action from environment
def compute_action(env_name, action):
    if env_name == 'CartPole-v1':
        action =  torch.sigmoid(action)
        return int(torch.round(action))
    elif env_name == 'MountainCar-v0':
        action = torch.relu(action)
        action = int(torch.round(action))
        return min(2, action)
    elif env_name == 'LunarLander-v3':
        action = torch.relu(action)
        action = int(torch.round(action))
        return min(3, action)
    
def get_model():
    if MODEL == 'static':
        model = StaticNN(input_size=env.observation_space.shape[0], output_size=1, hidden_sizes=[128, 64])
        n_variables = model.get_n_weights()
    elif MODEL == 'abcd':
        model =  HebbianAbcdNN(input_size=env.observation_space.shape[0], output_size=1, hidden_sizes=[64], env_name=ENV)
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



# Example usage
if __name__ == "__main__":

    if SEED:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)

    env = gym.make(ENV, render_mode="human", max_episode_steps=MAX_EPISODE_STEPS)
    model, n_variables = get_model()
    print(f'n_variables = {n_variables}, Layers = {len(model.layers)}')

    # Instantiate the CMA-ES optimizer
    optimizer = CMA_ES(func=objective_function, dim=n_variables, sigma=SIGMA, popsize=POPULATION_SIZE)

    # Run optimization
    best_solution, best_score = optimizer.optimize(iterations=ITERATIONS, stop_condition=STOP_CONDITION, seed=SEED)
    print("Best solution:", best_solution)
    print("Best score:", best_score)

    # Show best solution
    total_reward = objective_function(best_solution, tries = EVAL_TRIES, show=True)
    print(-total_reward)

