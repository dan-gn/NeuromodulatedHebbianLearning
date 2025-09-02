"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
EVOLUTIONARY ALGORITHM
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor

class Individual:

    def __init__(self, n_variables, genotype = None, fitness = None):
        self.n_variables = n_variables
        self.genotype = genotype
        self.fitness = fitness
        self.initial_value_range = 5 # +- initial_value_range

    def random_initialise(self):
        self.genotype = np.random.uniform(-self.initial_value_range, self.initial_value_range, self.n_variables)

class EvolutionaryAlgorithm:

    def __init__(self, n_variables, max_iterations, population_size, max_stagnment, model_name, environment_name, tries, objective_function):
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
        self.objective_function = objective_function

    def init_best_individual(self):
        self.best_individual = Individual(self.n_variables)
        self.best_individual.fitness = float("inf")

    def initialise_population(self):
        population = [Individual(self.n_variables) for _ in range(self.population_size)]
        for i, member in enumerate(population):
            population[i].random_initialise()
            population[i].fitness = self.objective_function(member.genotype, seed = self.seed, model_name=self.model_name, environment_name=self.environment_name, tries=self.tries)
            if population[i].fitness < self.best_individual.fitness:
                self.best_individual.genotype = population[i].genotype
                self.best_individual.fitness = population[i].fitness
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
            fitness_g1 = self.objective_function(mutated_g1, seed = self.seed, model_name=self.model_name, environment_name=self.environment_name, tries=self.tries)
            offspring.append(Individual(self.n_variables, genotype=mutated_g1, fitness=fitness_g1))
            mutated_g2 = self.mutate(genotype2)
            fitness_g2 = self.objective_function(mutated_g2, seed = self.seed, model_name=self.model_name, environment_name=self.environment_name, tries=self.tries)
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
            fitnesses = list(executor.map(lambda g: self.objective_function(g, seed=self.seed, model_name=self.model_name, environment_name=self.environment_name, tries=self.tries), offspring_genotypes))

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
            print(f'Iteration = {self.i}, Mean fitness = {np.mean([xi.fitness for xi in self.population])}, Best fitness = {self.best_individual.fitness}')
            if self.best_individual.fitness <= stop_criteria:
                print('Stop criteria achieved!')
                break
            self.record[self.i + 1] = self.best_individual.fitness
        return self.best_individual.genotype, self.best_individual.fitness