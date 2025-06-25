import numpy as np

# Rosenbrock function with dynamically changing minimum
def rosenbrock(x, t):
    """Dynamic Rosenbrock function with time-varying minimum"""
    a = 1.0
    b = 100.0
    x_shifted = x - np.sin(t)  # Dynamic shift of minimum over time
    noise = np.random.normal(0, 10)
    return ( (a - x_shifted[0])**2 + b * (x_shifted[1] - x_shifted[0]**2)**2 + noise, )

# Quadratic function with random noise
def quadratic(x, t):
    x = np.array(x)
    shift = np.sin(t)  # time-dependent shift (e.g., periodic)
    noise = np.random.normal(0, 1, size=x.shape)
    x_shifted = x - shift  # shift the minimum over time
    return (np.sum(x_shifted**2) + np.sum(noise), )

# Rastrigin function with dynamically changing parameters
def rastrigin(x, t):
    """Rastrigin function with moving optimum"""
    x = np.array(x)  # Convert input to numpy array
    A = 10 
    shift = np.sin(t)
    x_shifted = x - shift  # Moving optimum
    noise = np.random.normal(0, 1)
    return (A * len(x) + np.sum(x_shifted**2 - A * np.cos(2 * np.pi * x_shifted)) + noise, )
