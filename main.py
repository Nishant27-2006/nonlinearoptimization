import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Parameters for the optimization problem
np.random.seed(42)
n = 20
k = np.random.uniform(1, 10, size=n)
c = np.random.uniform(0.5, 2.0, size=n)
beta = 1e-3
S_max = 200
C_budget = 100
reg_param_initial = 0.01

# Data for plotting
objective_values = []
gradient_magnitudes = []

# Define the objective function with gradient tracking
def objective_function(x, reg_param=reg_param_initial):
    # Compute objective value
    value = np.sum(k * x**2 - c * x + beta / (x + 1e-8)) + reg_param * np.linalg.norm(x, 1)
    objective_values.append(value)  # Track objective values

    # Compute gradient (finite difference approximation)
    epsilon = 1e-8
    gradient = np.zeros_like(x)
    for i in range(len(x)):
        x_eps = np.copy(x)
        x_eps[i] += epsilon
        gradient[i] = (objective_function_no_track(x_eps, reg_param) - value) / epsilon

    gradient_magnitudes.append(np.linalg.norm(gradient))  # Track gradient magnitude
    return value

# Auxiliary function to avoid infinite recursion during gradient computation
def objective_function_no_track(x, reg_param=reg_param_initial):
    return np.sum(k * x**2 - c * x + beta / (x + 1e-8)) + reg_param * np.linalg.norm(x, 1)

# Nonlinear constraint 1: Stress limit
def stress_constraint(x):
    return S_max - np.sum(x**2)

# Nonlinear constraint 2: Budget constraint
def budget_constraint(x):
    return C_budget - np.sum(c * x)

# Create the nonlinear constraints
nlc1 = NonlinearConstraint(stress_constraint, 0, np.inf)
nlc2 = NonlinearConstraint(budget_constraint, 0, np.inf)

# Bounds for the variables (non-negativity)
bounds = Bounds(0, np.inf)

# Initial guess for optimization variables
x0 = np.random.uniform(0.1, 1.0, size=n)

# Enhanced solver options
solver_options = {
    'verbose': 3,
    'maxiter': 1000,
    'gtol': 1e-8
}

# Track execution time
start_time = time.time()

# Perform the optimization
result = minimize(
    objective_function,
    x0,
    method='trust-constr',
    constraints=[nlc1, nlc2],
    bounds=bounds,
    options=solver_options
)

# Track execution time
execution_time = time.time() - start_time

# Custom Hessian approximation
def compute_hessian_approx(x):
    hessian = np.zeros((len(x), len(x)))
    epsilon = 1e-4
    for i in range(len(x)):
        for j in range(len(x)):
            x_ij_plus = np.copy(x)
            x_ij_plus[i] += epsilon
            x_ij_plus[j] += epsilon
            f_ij_plus = objective_function_no_track(x_ij_plus)

            x_ij_minus = np.copy(x)
            x_ij_minus[i] -= epsilon
            x_ij_minus[j] -= epsilon
            f_ij_minus = objective_function_no_track(x_ij_minus)

            hessian[i, j] = (f_ij_plus - 2 * result.fun + f_ij_minus) / (epsilon ** 2)
    return hessian

hessian_approx = compute_hessian_approx(result.x)

# Generate Visualizations

# 1. Convergence Plot
plt.figure(figsize=(8, 6))
plt.plot(objective_values, marker='o', label="Objective Value")
plt.title("Convergence of Objective Function")
plt.xlabel("Iteration")
plt.ylabel("Objective Function Value")
plt.grid(True)
plt.legend()
plt.savefig("convergence_plot.png")
plt.show()

# 2. Gradient Magnitude Plot
plt.figure(figsize=(8, 6))
plt.plot(gradient_magnitudes, marker='x', label="Gradient Magnitude")
plt.title("Gradient Magnitude Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Gradient Magnitude")
plt.grid(True)
plt.legend()
plt.savefig("gradient_magnitude_plot.png")
plt.show()

# 3. Hessian Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(hessian_approx, cmap="coolwarm", annot=False, square=True)
plt.title("Heatmap of Hessian Matrix at Solution")
plt.savefig("hessian_heatmap.png")
plt.show()

# 4. Variable Contribution Bar Plot
variable_contribution = k * result.x**2 - c * result.x + beta / (result.x + 1e-8)
plt.figure(figsize=(10, 6))
plt.bar(range(len(variable_contribution)), variable_contribution, color="skyblue")
plt.title("Variable Contribution to Objective Function")
plt.xlabel("Variable Index")
plt.ylabel("Contribution")
plt.grid(axis='y')
plt.savefig("variable_contribution_plot.png")
plt.show()
