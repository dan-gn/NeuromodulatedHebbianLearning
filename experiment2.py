import numpy as np
import torch 
import gymnasium as gym

from Algorithms.hebbian_learning import StaticNN
from Algorithms.cma_es import CMA_ES

# Environment selection
ENV_NUMBER = 0

ENVIRONMENTS = []
ENVIRONMENTS.append('CartPole-v1')
ENVIRONMENTS.append('MountainCar-v0')
ENVIRONMENTS.append('LunarLander-v3')

# Neural Network parameters
HIDDEN_LAYERS = 2
HIDDEN_SIZE = 64

# Optimisation parameters
MAX_EPISODE_STEPS = 1000
TRIES = 3
SIGMA = 0.5
POPULATION_SIZE = 100
ITERATIONS = 250

# Evaluation parameters
EVAL_TRIES = 3

# Environment parameters
ENV = ENVIRONMENTS[ENV_NUMBER]
if ENV == 'CartPole-v1':
    STOP_CONDITION = -MAX_EPISODE_STEPS * TRIES
elif ENV == 'MountainCar-v0':
    STOP_CONDITION = -50 * TRIES
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

# Objective function
def objective_function(x, tries = TRIES):
    env = gym.make(ENV, max_episode_steps=MAX_EPISODE_STEPS)
    model = StaticNN(input_size=env.observation_space.shape[0], output_size=1, hidden_size=HIDDEN_SIZE, hidden_layers=HIDDEN_LAYERS)
    model.update_weights(torch.Tensor(x))
    result = np.zeros(tries)
    action_count = np.zeros(env.action_space.n)
    for i in range(tries):
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

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(ENV, render_mode="human", max_episode_steps=MAX_EPISODE_STEPS)
    model = StaticNN(input_size=env.observation_space.shape[0], output_size=1, hidden_size=HIDDEN_SIZE, hidden_layers=HIDDEN_LAYERS)
    n_variables = model.get_n_weights()
    print(f'n_variables = {n_variables}, Layers = {len(model.hidden)}')

    # Instantiate the CMA-ES optimizer
    optimizer = CMA_ES(func=objective_function, dim=n_variables, sigma=SIGMA, popsize=POPULATION_SIZE)

    # Run optimization
    best_solution, best_score = optimizer.optimize(iterations=ITERATIONS, stop_condition=STOP_CONDITION)
    print("Best solution:", best_solution)
    print("Best score:", best_score)

    # Show best solution
    model.update_weights(torch.Tensor(best_solution))
    for _ in range(EVAL_TRIES):
        observation, info = env.reset()
        total_reward = 0
        episode_over = False
        while not episode_over:
            action = model.forward(torch.Tensor(observation))
            action = compute_action(ENV, action)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            total_reward += reward
    env.close()
    print(-total_reward)

