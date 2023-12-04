import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the objective functions
def function1(x, y):
    "First function f1(x, y) = x^2 + y^2 + 10sin(x) + 10cos(x)"
    return x**2 + y**2 + 10*np.sin(x) + 10*np.cos(y)

def function2(x, y):
    "Second function f2(x,y) = x^2 + y^2"
    return x**2 - y**2

# Define the gradient of the objective functions
def gradient1(x, y):
    df_dx = 2*x + 10*np.cos(x)
    df_dy = 2*y - 10*np.sin(y)
    return np.array([df_dx, df_dy])

def gradient2(x, y):
    df_dx = 2*x
    df_dy = -2*y
    return np.array([df_dx, df_dy])

# Gradient Descent Optimization
def gradient_descent(initial_point, learning_rate, num_iterations, objective_function, gradient_function):
    history = [initial_point]

    for _ in range(num_iterations):
        current_point = history[-1]
        gradient = gradient_function(*current_point)
        update = learning_rate * gradient
        new_point = current_point - update
        history.append(new_point)

    return np.array(history)

# Visualization function
def plot_optimization_surface(objective_function, title):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(X, Y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Objective Function Value')

    plt.show()

# Plot the optimization process
def plot_optimization_process(history, objective_function, title):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(X, Y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.scatter(history[:, 0], history[:, 1], [objective_function(*point) for point in history], color='red', marker='o', label='Optimization Path')
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Objective Function Value')

    plt.legend()
    plt.show()

def optimization(function, gradient, learning_rates, initial_points, num_iters):
    """
    Implement gradient descent process
    """
    for initial_point in initial_points:
        for learning_rate in learning_rates:
            history = gradient_descent(np.array(initial_point), learning_rate, num_iters, function, gradient)
            plot_optimization_process(history, function, f'Gradient Descent Optimization \nInitial Point: {initial_point}, Learning Rate: {learning_rate}')

if __name__ == "__main__":
    # Define information
    initial_points = [(5, 5), (-5, -5), (8, -8)]
    learning_rates = [0.01, 0.1, 0.5]
    num_iterations = 100

    # Gradient descent for function 1
    optimization(function=function1, gradient=gradient1, learning_rates=learning_rates, initial_points=initial_points, num_iters=num_iterations)

    # Gradient descent for function 2
    optimization(function=function2, gradient=gradient2, learning_rates=learning_rates, initial_points=initial_points, num_iters=num_iterations)
