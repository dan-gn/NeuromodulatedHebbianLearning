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
# from Optimisation.ea import EvolutionaryAlgorithm

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

# input_filename = 'output_MountainCar-v0_static_2025-01-27_21-10-50.pkl'
# input_filename = 'exp4_output_ea_CartPole-v1_neuromodulated_hb_seed-1_time-2025-03-23_16-47-04.pkl'
# input_filename = 'exp4_output_ea_MountainCar-v0_neuromodulated_hb_seed-2_time-2025-03-25_11-15-39.pkl'
# input_filename = 'exp4_output_ea_Acrobot-v1_neuromodulated_hb_seed-1_time-2025-03-28_20-51-31.pkl'
input_filename = 'exp4_output_ea_LunarLander-v3_static_seed-0_time-2025-08-05_21-40-21_lambda_0-05.pkl'

# Model options
MODEL_NUMBER = 1
MODELS = []
MODELS.append('abcd')
MODELS.append('neuromodulated_hb')
MODELS.append('static')

# Environment options
ENV_NUMBER = 2
ENVIRONMENTS = []
ENVIRONMENTS.append('MountainCar-v0')
ENVIRONMENTS.append('LunarLander-v3')
ENVIRONMENTS.append('CartPole-v1')
ENVIRONMENTS.append('Acrobot-v1')

# Optimisation parameters
POPULATION_SIZE = 50
ITERATIONS = 1000
EVALUATIONS = POPULATION_SIZE * ITERATIONS
TRIES = 10
MAX_STAGNMENT = 50
LAMBDA_DECAY = 0.05

# Evaluation parameters
SHOW_BEST = False    # Runs the best solution for EVAL_TRIES
EVAL_TRIES = 100

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
        # return int(nn.functional.hardtanh(action, 0, 2))
        probs = F.softmax(action, dim=0)  
        return int(probs.argmax())
    elif env_name == 'LunarLander-v3':
        # return int(nn.functional.hardtanh(action, 0, 3))
        probs = F.softmax(action, dim=0)  
        return int(probs.argmax())

# Get the model and the number of variables
def get_model(output_size, model_name, env, env_name, lambda_value):
    if model_name == 'static':
        model = StaticNN(input_size=env.observation_space.shape[0], output_size=output_size, hidden_sizes=HIDDEN_SIZES)
        n_variables = model.get_n_weights()
    elif model_name == 'abcd':
        model =  HebbianAbcdNN(input_size=env.observation_space.shape[0], output_size=output_size, hidden_sizes=HIDDEN_SIZES, env_name=env_name)
        n_variables = model.get_n_weights() * 5
    elif model_name == 'neuromodulated_hb':
        model =  TimeBasedNeuromodulatedHebbianNN(input_size=env.observation_space.shape[0], output_size=output_size, hidden_sizes=HIDDEN_SIZES, env_name=env_name, lambda_decay=lambda_value)
        n_variables = model.get_n_weights() * 5
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
def objective_function(x, model_name = MODEL, environment_name = ENV, tries = TRIES, show=False, seed = None, max_episode_steps = MAX_EPISODE_STEPS, lambda_value=LAMBDA_DECAY):
    max_episode_steps, STOP_CONDITION, HIDDEN_SIZES = set_model_and_environment_parameters(environment_name, model_name)
    if show:
        tmp_env = gym.make(environment_name, render_mode="rgb_array", max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordVideo(env=tmp_env, name_prefix="test-video", video_folder='Experiments/Results/test_july/')

    else:
        env = gym.make(environment_name, max_episode_steps=max_episode_steps)
    output_size = get_output_size(environment_name)
    model, n_variables = get_model(output_size, model_name, env, environment_name, lambda_value)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    if model_name == 'static':
        model.update_weights(torch.Tensor(x))
    elif model_name == 'abcd' or model_name == 'neuromodulated_hb':
        model.update_weights(torch.rand(n_variables))
        model.update_hebbian(torch.Tensor(x))
    result = np.zeros(tries)
    action_record = []
    with torch.no_grad():
        for i in range(tries):
            if model_name == 'abcd' or model_name == 'neuromodulated_hb':
                model.update_weights(torch.rand(n_variables))
            if seed is not None:
                observation, info = env.reset(seed=seed+i)
            else:
                observation, info = env.reset()
            episode_over = False
            # env.start_recording(video_name=f'{environment_name}_{model_name}_seed{seed+i}')
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
EVOLUTIONARY ALGORITHM
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
import math
from concurrent.futures import ProcessPoolExecutor

class Individual:

    def __init__(self, n_variables, genotype = None, fitness = None):
        self.n_variables = n_variables
        self.genotype = genotype
        self.fitness = fitness
        self.fitness_test = None
        self.initial_value_range = 5 # +- initial_value_range

    def random_initialise(self):
        self.genotype = np.random.uniform(-self.initial_value_range, self.initial_value_range, self.n_variables)

class EvolutionaryAlgorithm:

    def __init__(self, n_variables, max_iterations, population_size, max_stagnment, model_name, environment_name, tries, lambda_value):
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.n_variables = n_variables
        self.mutation_probability = 1 / n_variables
        self.mutation_eta = 5
        self.sbx_eta = 5
        self.elitism_proportion = 0.1
        self.elitism_index = int(self.elitism_proportion * self.population_size)
        self.init_best_individual()
        self.max_stagnment = max_stagnment
        self.model_name = model_name
        self.environment_name = environment_name
        self.tries = tries
        self.lambda_value = lambda_value

    def init_best_individual(self):
        self.best_individual = Individual(self.n_variables)
        self.best_individual.fitness = float("inf")

    def initialise_population(self):
        population = [Individual(self.n_variables) for _ in range(self.population_size)]
        for i, member in enumerate(population):
            population[i].random_initialise()
            population[i].fitness = objective_function(member.genotype, seed = self.seed, model_name=self.model_name, environment_name=self.environment_name, tries=self.tries, lambda_value=self.lambda_value)
            if population[i].fitness < self.best_individual.fitness:
                self.best_individual.genotype = population[i].genotype
                self.best_individual.fitness = population[i].fitness
                self.best_individual.fitness_test = objective_function(self.best_individual.genotype, seed = 1996, model_name=self.model_name, environment_name=self.environment_name, tries=100, lambda_value=self.lambda_value) 
        return population

    def roulette_wheel(self, p):
        r = np.random.uniform(0, 1) * sum(p)	
        q = np.cumsum(p)
        return next(idx for idx, value in enumerate(q) if value >= r)

    def tournament_selection(self, n_competitors=2):
        all_indexes = list(range(self.population_size))
        parents = []
        for i in range(2):
            draw = np.random.permutation(all_indexes)
            competitors = draw[:n_competitors]
            winner = self.roulette_wheel(self.probs[competitors])
            parents.append(competitors[winner])
            all_indexes.remove(parents[i])
        return [self.population[parents[0]], self.population[parents[1]]]

    # Get parent selection probabilities
    def compute_parent_selection_prob(self, beta=1):
        # Get an array of all cost of current population, add acceptance criteria value
        # and divide by the mean of the array to avoid overflow while computing exponential
        fitness = np.array([member.fitness for member in self.population]) 
        mean_fitness = abs(np.mean(fitness))
        if mean_fitness != 0 and mean_fitness != math.inf:
            fitness /= mean_fitness
        return np.exp(-beta * fitness)
    
    def parent_selection(self):
        self.probs = self.compute_parent_selection_prob()
        parents = [self.tournament_selection() for _ in range(int(self.population_size/2))]
        # return np.array(parents).reshape(-1, 2).tolist()
        return parents
    
    def sbx(self, parents):
        # Ensure parents are numpy arrays
        parent1 = np.array(parents[0].genotype)
        parent2 = np.array(parents[1].genotype)
        # Random numbers for each dimension
        rand = np.random.rand(len(parent1))
        # Compute beta values for each dimension
        beta = np.empty_like(rand)
        mask = rand <= 0.5
        beta[mask] = (2 * rand[mask]) ** (1 / (self.sbx_eta + 1))
        beta[~mask] = (1 / (2 * (1 - rand[~mask]))) ** (1 / (self.sbx_eta + 1))
        # Create offspring
        offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
        return offspring1, offspring2

    def polynomial_muatation(self, x):
        r = np.random.uniform(0, 1)
        if r < 0.5:
            delta = (2*r) ** (1 / (self.mutation_eta+1)) - 1
        else:
            delta = 1 - (2 * (1-r)) ** (1 / (self.mutation_eta+1))
        return x + delta

    def elitism(self, offspring):
        self.population = sorted(self.population, key=lambda x: x.fitness)
        offspring = sorted(offspring, key=lambda x: x.fitness)
        self.population[self.elitism_index:] = offspring[:-self.elitism_index]

        if offspring[0].fitness < self.best_individual.fitness:
            self.best_individual.genotype = offspring[0].genotype
            self.best_individual.fitness = offspring[0].fitness
            self.best_individual.fitness_test = objective_function(self.best_individual.genotype, seed = 1996, model_name=self.model_name, environment_name=self.environment_name, tries=100, lambda_value=self.lambda_value) 
            self.stagnment_iterations = -1
        self.stagnment_iterations += 1

    def mutate(self, genotype):
        genotype = np.array(genotype)  # ensure it's a NumPy array
        random_values = np.random.uniform(0, 1, size=genotype.shape)
        mutation_mask = random_values <= self.mutation_probability
        for idx in np.where(mutation_mask)[0]:
            genotype[idx] = self.polynomial_muatation(genotype[idx])
        return genotype

    def crossover_and_mutation(self, parents):
        # offspring = [Individual(self.n_variables) for _ in range(self.population_size)]
        # for i, p in enumerate(parents):
        #     genotype1, genotype2 = self.sbx(p)
        #     offspring[i*2].genotype = self.mutate(genotype1)
        #     offspring[i*2].fitness = self.evaluate(offspring[i*2].genotype, seed = self.seed)
        #     offspring[i*2 + 1].genotype = self.mutate(genotype2)
        #     offspring[i*2 + 1].fitness = self.evaluate(offspring[i*2 + 1].genotype, seed = self.seed)
        # return offspring
        offspring = []
        for p in parents:
            genotype1, genotype2 = self.sbx(p)
            mutated_g1 = self.mutate(genotype1)
            fitness_g1 = objective_function(mutated_g1, seed = self.seed, model_name=self.model_name, environment_name=self.environment_name, tries=self.tries, lambda_value=self.lambda_value)
            offspring.append(Individual(self.n_variables, genotype=mutated_g1, fitness=fitness_g1))
            mutated_g2 = self.mutate(genotype2)
            fitness_g2 = objective_function(mutated_g2, seed = self.seed, model_name=self.model_name, environment_name=self.environment_name, tries=self.tries, lambda_value=self.lambda_value)
            offspring.append(Individual(self.n_variables, genotype=mutated_g2, fitness=fitness_g2))
        return offspring
    
    def parallel_crossover_and_mutation(self, parents):
        offspring_genotypes = []
        for p in parents:
            genotype1, genotype2 = self.sbx(p)
            offspring_genotypes.append(self.mutate(genotype1))
            offspring_genotypes.append(self.mutate(genotype2))

        # Parallel evaluation of all genotypes
        with ProcessPoolExecutor() as executor:
            fitnesses = list(executor.map(lambda g: objective_function(g, seed=self.seed, model_name=self.model_name, environment_name=self.environment_name, tries=self.tries, lambda_value=self.lambda_value), offspring_genotypes))

        # Build offspring individuals
        offspring = [
            Individual(self.n_variables, genotype=g, fitness=f)
            for g, f in zip(offspring_genotypes, fitnesses)
        ]

        return offspring

    def update_population(self):
        parents = self.parent_selection()
        # offspring = self.crossover(parents)
        # offspring = self.mutation(offspring)
        # start_time = time.time()
        offspring = self.crossover_and_mutation(parents)
        # print(f'Normal time = {time.time() - start_time}')
        # start_time = time.time()
        # offspring = self.parallel_crossover_and_mutation(parents)
        # print(f'Parallel time = {time.time() - start_time}')
        self.elitism(offspring)

    def run(self, stop_criteria, seed):
        self.record = np.zeros(self.max_iterations + 1)
        self.seed = seed
        self.stagnment_iterations = 0
        self.population = self.initialise_population()
        for self.i in range(self.max_iterations):
            self.record[self.i] = self.best_individual.fitness
            if self.stagnment_iterations >= self.max_stagnment:
                print('Restart population!')
                self.stagnment_iterations = -1
                self.population = self.initialise_population()
                self.population = sorted(self.population, key=lambda x: x.fitness)
                self.population[-1].genotype = self.best_individual.genotype
                self.population[-1].fitness = self.best_individual.fitness
            else:
                self.update_population()
            if self.i % 25 == 0:
                print(f'Iteration = {self.i}, Mean fitness = {np.mean([xi.fitness for xi in self.population])}, Best fitness = {self.best_individual.fitness}, Best fitness testing = {self.best_individual.fitness_test}')
            if self.best_individual.fitness <= stop_criteria:
                print('Stop criteria achieved!')
                break
            self.record[self.i + 1] = self.best_individual.fitness
        return self.best_individual.genotype, self.best_individual.fitness


lambda_exp = [x/2 for x in range(9)]
lambdas = [10**(-x) for x in lambda_exp]

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MAIN FUNCTION
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":

    MAX_EPISODE_STEPS, STOP_CONDITION, HIDDEN_SIZES = set_model_and_environment_parameters(ENV, MODEL)

    for i, lambd in enumerate(lambdas):
        for seed in range(15, 30):

            SEED = seed

            if SEED is not None:
                print(f'Seed set to {SEED}.')
                torch.manual_seed(SEED)
                torch.cuda.manual_seed(SEED)
                np.random.seed(SEED)
                random.seed(SEED)

            env = gym.make(ENV, render_mode="human", max_episode_steps=MAX_EPISODE_STEPS)
            output_size = get_output_size(ENV)
            model, n_variables = get_model(output_size, MODEL, env, ENV, lambd)
            print(f'MODEL = {MODEL}, ENVIRONMENT = {ENV}')
            print(f'hidden = {HIDDEN_SIZES}, n_variables = {n_variables}, Layers = {len(model.layers)}, Stopping criteria = {STOP_CONDITION}')

            # Instantiate the CMA-ES optimizer
            optimizer = EvolutionaryAlgorithm(n_variables=n_variables, max_iterations=ITERATIONS, population_size=POPULATION_SIZE, max_stagnment=MAX_STAGNMENT, model_name=MODEL, environment_name=ENV, tries=TRIES, lambda_value=lambd)
            start_time = time.time()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f'Optimisation started at {timestamp}.')

            if READ_DATA:
                with open(f'Experiments/Results/test_july/{input_filename}', 'rb') as file:
                    data = pickle.load(file)
                best_solution = data['best_solution']
                best_score = data['best_score']
            else:
                # Run optimization
                best_solution, best_score = optimizer.run(STOP_CONDITION, SEED)

            
            print("Best solution:", best_solution)
            print("Best score:", best_score)

            # Show best solution
            # total_reward = objective_function(best_solution, tries = EVAL_TRIES, show=SHOW_BEST, seed=1996, model_name=MODEL, environment_name=ENV)
            total_reward = objective_function(best_solution, tries = EVAL_TRIES, show=SHOW_BEST, seed=0, model_name=MODEL, environment_name=ENV)
            print(f'Evaluation total reward {total_reward}')

            # Store experiment data
            if STORE_DATA and not READ_DATA:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_filename = f'Experiments/Results/test_sept/exp7_output_ea_{ENV}_{MODEL}_seed-{seed}_time-{timestamp}_lambda_{i}.pkl'
                output = {
                    'best_solution': best_solution,
                    'best_score': best_score,
                    'evaluation_score' : total_reward,
                    'hidden_size': HIDDEN_SIZES,
                    'population_size': POPULATION_SIZE,
                    'max_iterations': ITERATIONS,
                    'n_iterations' : optimizer.i,
                    'record' : optimizer.record, 
                    'lambda_decay' : lambd,
                    'lambda_exp' : lambda_exp[i],
                    'seed': SEED
                    }
                with open(output_filename, 'wb') as file:
                    pickle.dump(output, file)

                log_file = f'Experiments/Results/test_sept/experiments_log.csv'
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
                    'n_variables' : n_variables,
                    'max_stagnment' : MAX_STAGNMENT,
                    'lambda_decay' : lambd,
                    'lambda_exp' : lambda_exp[i],
                    'tries' : TRIES,
                    'evaluation_score' : total_reward,
                    'eval_tries' : EVAL_TRIES,
                    'time' : time.time() - start_time
                    }
                append_line_to_csv(log_file, new_line)


