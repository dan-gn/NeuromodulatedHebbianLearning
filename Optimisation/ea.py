import numpy as np
import math

class Individual:

    def __init__(self, n_variables):
        self.n_variables = n_variables
        self.genotype = None
        self.fitness = None
        self.initial_value_range = 5 # +- initial_value_range

    def random_initialise(self):
        self.genotype = np.random.uniform(-self.initial_value_range, self.initial_value_range, self.n_variables)

class EvolutionaryAlgorithm:

    def __init__(self, n_variables, objective_function, max_iterations, population_size, max_stagnment):
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.n_variables = n_variables
        self.evaluate = objective_function
        self.mutation_probability = 1 / n_variables
        self.mutation_eta = 5
        self.sbx_eta = 5
        self.elitism_proportion = 0.1
        self.elitism_index = int(self.elitism_proportion * self.population_size)
        self.init_best_individual()
        self.max_stagnment = max_stagnment

    def init_best_individual(self):
        self.best_individual = Individual(self.n_variables)
        self.best_individual.fitness = float("inf")

    def initialise_population(self):
        population = [Individual(self.n_variables) for _ in range(self.population_size)]
        for i, member in enumerate(population):
            population[i].random_initialise()
            population[i].fitness = self.evaluate(member.genotype, seed = self.seed)
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

    def crossover(self, parents):
        offspring = [Individual(self.n_variables) for _ in range(self.population_size)]
        for i, p in enumerate(parents):
            genotype1, genotype2 = self.sbx(p)
            offspring[i*2].genotype = genotype1
            offspring[i*2 + 1].genotype = genotype2
        return offspring

    def polynomial_muatation(self, x):
        r = np.random.uniform(0, 1)
        if r < 0.5:
            delta = (2*r) ** (1 / (self.mutation_eta+1)) - 1
        else:
            delta = 1 - (2 * (1-r)) ** (1 / (self.mutation_eta+1))
        return x + delta

    def mutation(self, population):
        for i, member in enumerate(population):
            for j, gen in enumerate(member.genotype):
                r = np.random.uniform(0, 1)
                if r <= self.mutation_probability:
                    population[i].genotype[j] = self.polynomial_muatation(gen)
        return population
    
    def evaluate_population(self, population):
        for i, member in enumerate(population):
            population[i].fitness = self.evaluate(member.genotype, seed = self.seed)
            if population[i].fitness < self.best_individual.fitness:
                self.best_individual.genotype = population[i].genotype
                self.best_individual.fitness = population[i].fitness
                self.stagnment_iterations = -1
        self.stagnment_iterations += 1
        return population

    def elitism(self, offspring):
        self.population = sorted(self.population, key=lambda x: x.fitness)
        offspring = sorted(offspring, key=lambda x: x.fitness)
        self.population[self.elitism_index:] = offspring[:-self.elitism_index]

    def update_population(self):
        parents = self.parent_selection()
        offspring = self.crossover(parents)
        offspring = self.mutation(offspring)
        offspring = self.evaluate_population(offspring)
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


# Sphere function (objective function to minimize)
def sphere_function(x):
    return np.sum(x**2)

# Himmelblau function
def himmelblau_function(z):
    x = z[0]
    y = z[1]
    """
    Compute the Himmelblau function.

    Parameters:
    - x: x-coordinate (float or numpy array).
    - y: y-coordinate (float or numpy array).

    Returns:
    - Function value at (x, y).
    """
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Rastrigin Function
def rastrigin_function(x):
    """
    Compute the Rastrigin function.

    Parameters:
    - x: Input vector (numpy array).

    Returns:
    - Function value at x.
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

if __name__ == "__main__":
    # algorithm = EvolutionaryAlgorithm(objective_function=sphere_function, n_variables=10)
    # algorithm = EvolutionaryAlgorithm(objective_function=himmelblau_function, n_variables=2)
    algorithm = EvolutionaryAlgorithm(objective_function=rastrigin_function, n_variables=10, max_iterations=200, population_size=50)

    algorithm.run(0, None)
