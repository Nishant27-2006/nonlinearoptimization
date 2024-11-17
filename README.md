# Dynamic Regularization in Trust-Region Optimization for Nonlinear Constraints

This repository contains a Python implementation of a robust optimization framework utilizing trust-region methods, dynamic regularization, and sensitivity analysis for solving high-dimensional nonlinear constrained optimization problems.

## Features

- **Optimization Framework**: Implements a trust-region optimization algorithm with dynamic regularization to promote sparsity and feasibility.
- **Nonlinear Constraints**: Includes stress and budget constraints to mimic real-world scenarios.
- **Sensitivity Analysis**: Evaluates gradient magnitudes and approximates the Hessian matrix for detailed insights.
- **Advanced Visualizations**: Generates four types of plots to understand the optimization process and results:
  1. **Convergence Plot**: Tracks the objective function value over iterations.
  2. **Gradient Magnitude Plot**: Illustrates the decay of gradient magnitudes over iterations.
  3. **Hessian Heatmap**: Visualizes the curvature and structural relationships among decision variables.
  4. **Variable Contribution Plot**: Highlights the impact of each variable on the objective function.

---

## Installation

To run this script, ensure you have Python 3.7 or later installed along with the following dependencies:

- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`

You can install the required packages using pip:

pip install numpy scipy matplotlib seaborn
Usage
Clone this repository:
bash
Copy code
git clone <repository-url>
cd <repository-folder>
Run the main.py script:

python main.py
Input Parameters
n: Number of decision variables (default: 20).
k: Elasticity constants for the objective function (randomized).
c: Material resistance constants for the objective function (randomized).
beta: Regularization term (default: 1e-3).
S_max: Maximum allowable stress (default: 200).
C_budget: Budget constraint (default: 100).
These parameters can be adjusted directly within the main.py script.

Output
The script outputs the following:

Optimal Solution: Prints the optimized decision variables and the minimum value of the objective function.
Execution Time: Displays the total time taken for optimization.
Visualizations: Saves the following plots in the working directory:
convergence_plot.png
gradient_magnitude_plot.png
hessian_heatmap.png
variable_contribution_plot.png
Visualizations
1. Convergence Plot
This plot shows the rapid reduction of the objective function value over iterations, demonstrating the efficiency of the optimization framework.

2. Gradient Magnitude Plot
Illustrates the decay of gradient magnitudes over iterations, verifying the robustness and convergence of the algorithm.

3. Hessian Heatmap
A heatmap visualizing the curvature and dependencies among decision variables at the optimal solution.

4. Variable Contribution Plot
A bar chart showing the contribution of each decision variable to the objective function, highlighting influential variables.

Customization
Initial Guess: Modify x0 to change the starting point of the optimization.
Solver Options: Adjust solver_options to change optimization settings (e.g., maximum iterations, gradient tolerance).
Visualization Style: Customize plot styles by modifying matplotlib or seaborn parameters.
Example Run
After running the script, you should see the following outputs:

Optimal solution:
less
Copy code
Optimal solution (x): [ ... values ... ]
Minimum objective value: ...
Execution time: 0.36 seconds
Visualizations saved as PNG files in the working directory.
License
This project is licensed under the MIT License. Feel free to use and modify the code for your research or projects.

Acknowledgments
This implementation leverages the scipy.optimize module for the trust-region algorithm and uses matplotlib and seaborn for advanced visualizations.

For questions or contributions, feel free to create an issue or submit a pull request.


This `README.md` is structured to provide a clear overview of the project, its functionality, and how to use it. Let me know if you need further modifications!





