import numpy as np
import torch
import gymnasium as gym

from Algorithms.hebbian_learning import HebbianABCDLayer

def objective_function(x, tries = 3):
    model = HebbianABCDLayer(4, 1)
    model.update_weights(x)
    result = np.zeros(tries)
    env = gym.make("CartPole-v1")
    for i in range(tries):
        observation, info = env.reset()
        episode_over = False
        while not episode_over:
            action = model.forward(torch.Tensor(observation))
            action = int(torch.round(action))
            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            result[i] += reward
    return np.sum(result)


if __name__ == '__main__':
     
    model = HebbianABCDLayer(4, 1)

    env = gym.make("CartPole-v1", render_mode="human")

    # env = gym.make("LunarLander-v3", render_mode="human")
    observation, info = env.reset()

    total_reward = 0
    episode_over = False
    while not episode_over:
        # action = env.action_space.sample()  # agent policy that uses the observation and info
        action = model.forward(torch.Tensor(observation))
        action = int(torch.round(action))
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        total_reward += reward

    env.close()

    print(total_reward)

    x = torch.rand(1,4)
    print(x)
    print(objective_function(x))