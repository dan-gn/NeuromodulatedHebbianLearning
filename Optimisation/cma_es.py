import numpy as np

class CMA_ES:
    def __init__(self, func, dim, sigma=0.5, popsize=50):
        self.func = func  # Objective function to minimize
        self.dim = dim  # Dimensionality of the search space
        self.initial_sigma = sigma  # Initial step size
        self.popsize = popsize  # Population size
        self.restart_count = -1
        self.restart()
        self.best_solution = None
        self.best_score = float("inf")

    def restart(self):
        self.restart_count += 1
        self.mean = np.zeros(self.dim)  # Initial mean
        self.cov = np.eye(self.dim, dtype=np.float16)  # Initial covariance matrix
        self.sigma = self.initial_sigma
        if self.restart_count >= 1:
            self.mean = self.best_solution
    
    def sample_population(self):
        """Samples a population of candidates."""
        return np.random.multivariate_normal(self.mean, self.cov, self.popsize)
    
    def update_distribution(self, population, scores):
        """Updates the mean and covariance matrix."""
        # Sort by fitness
        sorted_indices = np.argsort(scores)
        population = population[sorted_indices]
        scores = scores[sorted_indices]
        
        # Update mean (weighted sum of top-performing individuals)
        weights = np.log(self.popsize + 1) - np.log(np.arange(1, self.popsize + 1))
        weights /= weights.sum()  # Normalize weights
        mean_new = np.sum(population[:self.popsize].T * weights, axis=1)
        
        # Update covariance matrix
        diff = population[:self.popsize] - mean_new
        cov_new = diff.T @ np.diag(weights) @ diff
        
        # Update step size (optional: scale by variance)
        step_size = self.sigma * np.sqrt(np.var(scores))
        
        self.mean = mean_new
        self.cov = cov_new
        self.sigma = step_size
        
        # Save the best solution
        if scores[0] < self.best_score:
            self.best_solution = population[0]
            self.best_score = scores[0]
    
    def optimize(self, iterations=100, stop_condition=None, seed=None, with_restarts=False):
        """Runs the optimization."""
        for self.i in range(iterations):
            # Sample population
            population = self.sample_population()
            
            # Evaluate the population
            scores = np.array([self.func(ind, seed=seed) for ind in population])
            
            # Update the distribution
            self.update_distribution(population, scores)

            print(f'Iteration = {self.i}, Mean score = {scores.mean()}, Best score = {self.best_score}, Sigma = {self.sigma}, Restarts = {self.restart_count}')

            if stop_condition is not None:
                if self.best_score <= stop_condition:
                    print('Stopping condition achieved.')
                    break

            # self.func(self.best_solution, tries=1, show=True)
            if with_restarts and (self.sigma < 1e-8 or self.sigma > 1e16):
                self.restart()
            
        return self.best_solution, self.best_score


# Example usage
if __name__ == "__main__":
    # Define a sample objective function (e.g., Sphere function)
    def objective_function(x):
        return np.sum(x**2)

    # Instantiate the CMA-ES optimizer
    optimizer = CMA_ES(func=objective_function, dim=5, sigma=0.5, popsize=20)

    # Run optimization
    best_solution, best_score = optimizer.optimize(iterations=200)
    print("Best solution:", best_solution)
    print("Best score:", best_score)