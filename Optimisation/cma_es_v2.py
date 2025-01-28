import numpy as np

class PureCMAES:
    def __init__(self, fitness_function, N=20, sigma=0.3, stopfitness=1e-10):
        self.fitness_function = fitness_function
        self.N = N
        self.sigma = sigma
        self.stopfitness = stopfitness
        # self.stopeval = int(1e3 * N ** 2)
        self.stopeval = 1000
        
        # Strategy parameter setting: Selection
        self.lambda_ = 4 + int(3 * np.log(N))
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        
        # Strategy parameter setting: Adaptation
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (N + 1)) - 1) + self.cs
        
        # Initialize dynamic strategy parameters
        self.xmean = np.random.rand(N)
        self.pc = np.zeros(N)
        self.ps = np.zeros(N)
        self.B = np.eye(N)
        self.D = np.ones(N)
        self.C = self.B @ np.diag(self.D ** 2) @ self.B.T
        self.invsqrtC = self.B @ np.diag(self.D ** -1) @ self.B.T
        self.eigeneval = 0
        self.chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
    
    def sample_population(self):
        # Generate and evaluate lambda offspring
        arx = np.zeros((self.N, self.lambda_))
        arfitness = np.zeros(self.lambda_)
        for k in range(self.lambda_):
            arx[:, k] = self.xmean + self.sigma * self.B @ (self.D * np.random.randn(self.N))
            arfitness[k] = self.fitness_function(arx[:, k])
            self.counteval += 1
        return arx, arfitness
    
    def update_distribution(self, arx, arfitness):
        # Sort by fitness and compute weighted mean
        arindex = np.argsort(arfitness)
        xold = self.xmean.copy()
        self.xmean = arx[:, arindex[:self.mu]] @ self.weights

        # Cumulation: Update evolution paths
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.invsqrtC @ (self.xmean - xold) / self.sigma
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lambda_)) / self.chiN) < (1.4 + 2 / (self.N + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.xmean - xold) / self.sigma

        # Adapt covariance matrix C
        artmp = (arx[:, arindex[:self.mu]] - xold[:, None]) / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * artmp @ np.diag(self.weights) @ artmp.T

        # Adapt step size sigma
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

        # Decomposition of C
        if self.counteval - self.eigeneval > self.lambda_ / (self.c1 + self.cmu) / self.N / 10:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D)
            self.invsqrtC = self.B @ np.diag(1 / self.D) @ self.B.T
        
        return arindex


    def optimize(self):
        self.counteval = 0
        while self.counteval < self.stopeval:
            # Generate and evaluate lambda offspring
            arx, arfitness = self.sample_population()

            arindex = self.update_distribution()
            
            print(f'Iteration = {self.counteval}, Best score = {arfitness[arindex[0]]}, Sigma = {self.sigma}')

            # Stopping criteria
            if arfitness[arindex[0]] <= self.stopfitness or max(self.D) > 1e7 * min(self.D):
                break
        
        return arx[:, arindex[0]], arfitness[arindex[0]]  # Return best solution