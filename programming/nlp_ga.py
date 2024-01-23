from sko.GA import GA
from sko.operators import ranking, selection, crossover, mutation

# New objective function:minimize
def objective(x):
    x1, x2,x3,x4,x5 = x
    return - (x1**2 + x2**2 + 3* x3**2 + 4*x4**2 + 2*x5**2 - 8*x1 - 2*x2 - 3*x3 - x4 - 2*x5)

# Constraint functions: constraints<=0
def constraint1(x):
    return x[0] + x[1] + x[2] + x[3] + x[4] - 400

def constraint2(x):
    return x[0] + 2*x[1] + 2*x[2] + x[3] + 6*x[4] - 800 

def constraint3(x):
    return 2*x[0] + x[1] + 6*x[2] - 200 

def constraint4(x):
    return x[2] + x[3] + 5*x[4] - 200 

# Combine all constraints
constraints_ueq = [constraint1, constraint2, constraint3, constraint4]

# bounds
lb = [0]*5
ub = [99]*5

# Run genetic algorithm
ga = GA(func=objective, n_dim=5, size_pop=100, max_iter=500,  prob_mut=0.001, precision=[1e-7] * 5,constraint_ueq=constraints_ueq, lb=lb, ub=ub)
ga.register(operator_name='selection', operator=selection.selection_roulette_2)
ga.register(operator_name='ranking', operator=ranking.ranking). \
register(operator_name='crossover', operator=crossover.crossover_2point_bit). \
register(operator_name='mutation', operator=mutation.mutation)

best_params, best_objective_value = ga.run()

# Output the results
print("Best Parameters:", best_params)
print("Best Objective Value:", -best_objective_value)
