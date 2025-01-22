from pymoo.core.termination import Termination

class ValueBasedTermination(Termination):
    def __init__(self, target_value, max_evals=None, max_iters=None, tol=1e-6):
        super().__init__()
        self.target_value = target_value
        self.max_evals = max_evals
        self.max_iters = max_iters
        self.tol = tol  # Tolerance for floating-point comparisons

    def _update(self, algorithm):
        best = algorithm.opt[0].F  # Best function value so far
        n_evals = algorithm.evaluator.n_eval  # Number of function evaluations
        n_iters = algorithm.n_iter  # Number of iterations
        
        # Check stopping conditions
        value_reached = best <= self.target_value + self.tol
        evals_exceeded = self.max_evals is not None and n_evals >= self.max_evals
        iters_exceeded = self.max_iters is not None and n_iters >= self.max_iters

        return value_reached or evals_exceeded or iters_exceeded

