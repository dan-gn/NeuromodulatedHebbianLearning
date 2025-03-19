import numpy as np

class CMA_ES:
    def __init__(self, objective_function, dim, population_size=None, initial_mean=None, initial_sigma=1.0):
        """
        Initialize the CMA-ES optimizer.

        Parameters:
        - objective_function: The function to minimize.
        - dim: Dimensionality of the problem.
        - population_size: Number of samples per generation (optional).
        - initial_mean: Initial mean of the distribution (optional).
        - initial_sigma: Initial step size (standard deviation).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.sigma = initial_sigma

        # Set population size (lambda)
        if population_size is None:
            self.population_size = 4 + int(3 * np.log(dim))  # Default heuristic
        else:
            self.population_size = population_size

        # Initialize mean
        if initial_mean is None:
            self.mean = np.zeros(dim)
        else:
            self.mean = np.array(initial_mean)

        # Initialize covariance matrix and evolution paths
        self.C = np.eye(dim)  # Covariance matrix
        self.pc = np.zeros(dim)  # Evolution path for C
        self.ps = np.zeros(dim)  # Evolution path for sigma

        # Strategy parameters
        self.mu = self.population_size // 2  # Number of parents
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)  # Normalize weights
        self.mueff = 1 / np.sum(self.weights**2)  # Effective mu

        # Adaptation parameters
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

    def sample_population(self):
        """Sample a new population from the current distribution."""
        self.population = np.random.multivariate_normal(self.mean, self.sigma**2 * self.C, self.population_size)

    def evaluate_population(self):
        """Evaluate the objective function for each individual in the population."""
        self.fitness = np.array([self.objective_function(ind) for ind in self.population])

    def update_distribution(self):
        """Update the mean, covariance matrix, and step size."""
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness)
        self.population = self.population[sorted_indices]
        self.fitness = self.fitness[sorted_indices]

        # Update mean
        old_mean = self.mean
        self.mean = np.sum(self.weights[:, np.newaxis] * self.population[:self.mu], axis=0)

        # Update evolution paths
        y = (self.mean - old_mean) / self.sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.dot(np.linalg.inv(np.linalg.cholesky(self.C)), y)

        # Update covariance matrix
        hs = 1 if np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * self.generation)) < 1.4 + 2 / (self.dim + 1) else 0
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * np.outer(self.pc, self.pc) + self.cmu * np.sum(self.weights[:, np.newaxis, np.newaxis] * np.array([np.outer(x, x) for x in (self.population[:self.mu] - old_mean) / self.sigma]), axis=0)

        # Update step size
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.mean(np.random.chisquare(self.dim, 1000)) - 1))

    def optimize(self, max_generations=1000, tol=1e-6):
        """Run the CMA-ES optimization."""
        for generation in range(max_generations):
            self.sample_population()
            self.evaluate_population()
            self.update_distribution()

            # Check for convergence
            if np.std(self.fitness) < tol:
                print(f"Converged at generation {generation}")
                break

        return self.mean, self.fitness[0]

# Example usage
if __name__ == "__main__":
    # Define the objective function (e.g., Sphere function)
    def sphere_function(x):
        return np.sum(x**2)

    # Initialize CMA-ES
    dim = 10  # Dimensionality of the problem
    cma_es = CMAES(sphere_function, dim, initial_sigma=1.0)

    # Run optimization
    best_solution, best_fitness = cma_es.optimize(max_generations=1000)

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")