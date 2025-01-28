import numpy as np

def frosenbrock(x):
    if len(x) < 2:
        raise ValueError("Dimension must be greater than one")
    return 100 * np.sum((x[:-1] ** 2 - x[1:]) ** 2) + np.sum((x[:-1] - 1) ** 2)

def purecmaes():
    # Initialization
    N = 20  # Problem dimension
    xmean = np.random.rand(N)  # Initial point
    sigma = 0.3  # Step size
    stopfitness = 1e-10  # Stopping criterion
    stopeval = int(1e3 * N ** 2)  # Maximum number of evaluations
    
    # Strategy parameter setting: Selection
    lambda_ = 4 + int(3 * np.log(N))  # Population size
    mu = lambda_ // 2  # Number of parents
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)  # Normalize weights
    mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)  # Variance-effectiveness
    
    # Strategy parameter setting: Adaptation
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs
    
    # Initialize dynamic strategy parameters
    pc = np.zeros(N)
    ps = np.zeros(N)
    B = np.eye(N)
    D = np.ones(N)
    C = B @ np.diag(D ** 2) @ B.T
    invsqrtC = B @ np.diag(D ** -1) @ B.T
    eigeneval = 0
    chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
    
    # Generation Loop
    counteval = 0
    while counteval < stopeval:
        # Generate and evaluate lambda offspring
        arx = np.zeros((N, lambda_))
        arfitness = np.zeros(lambda_)
        for k in range(lambda_):
            arx[:, k] = xmean + sigma * B @ (D * np.random.randn(N))
            arfitness[k] = frosenbrock(arx[:, k])
            counteval += 1
        
        # Sort by fitness and compute weighted mean
        arindex = np.argsort(arfitness)
        xold = xmean.copy()
        xmean = arx[:, arindex[:mu]] @ weights
        
        # Cumulation: Update evolution paths
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (xmean - xold) / sigma
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lambda_)) / chiN) < (1.4 + 2 / (N + 1))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
        
        # Adapt covariance matrix C
        artmp = (arx[:, arindex[:mu]] - xold[:, None]) / sigma
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp @ np.diag(weights) @ artmp.T
        
        # Adapt step size sigma
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        
        # Decomposition of C
        if counteval - eigeneval > lambda_ / (c1 + cmu) / N / 10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T
            D, B = np.linalg.eigh(C)
            D = np.sqrt(D)
            invsqrtC = B @ np.diag(1 / D) @ B.T
        
        # Stopping criteria
        if arfitness[arindex[0]] <= stopfitness or max(D) > 1e7 * min(D):
            break
    
    return arx[:, arindex[0]]  # Return best solution


if __name__ == "__main__":
    print(purecmaes())