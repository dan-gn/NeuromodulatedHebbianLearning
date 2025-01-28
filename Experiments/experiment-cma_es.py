import numpy as np

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("sphere")

algorithm = CMAES(
    x0=np.random.random(problem.n_var),
    # restarts=1,
    tolfun=1e-2,
    # tolx=1e-2,
    tolstagnation=10
    )

from pymoo.termination import get_termination
termination = get_termination('n_gen', 1000)
# termination += get_termination('n_eval', 2000)
# termination += get_termination('fmin', 0)

res = minimize(problem,
               algorithm,
               termination=termination,
               seed=1,
               verbose=True)

# Check the termination reason
print("Termination reason:", res)

# Check the optimal value found
print("Optimal value:", res)

# If you want to check the history of the optimization process:
print("History of generations:", res.history)

# Print the best solution found
print("Best solution found:", res.X)

# print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")

# res = minimize(problem,
#                algorithm,
#                ('n_iter', 10),
#                seed=1,
#                verbose=True)

# print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

# import cma

# xopt, es = cma.fmin2(cma.ff.rosen, 8 * [0], 0.5)