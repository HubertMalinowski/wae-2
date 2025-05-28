import numpy as np

# Funkcja Rosenbrocka z dynamicznie zmieniającym się minimum
def rosenbrock_dynamic(x, t):
    """Dynamic Rosenbrock function with time-varying minimum"""
    a = 1.0
    b = 100.0
    x_shifted = x - np.sin(t)  # Dynamic shift of minimum over time
    return (a - x_shifted[0])**2 + b * (x_shifted[1] - x_shifted[0]**2)**2,

# Funkcja kwadratowa z szumem losowym
def noisy_quadratic(x):
    """Quadratic function with random noise"""
    x = np.array(x)  # Convert input to numpy array
    noise = np.random.normal(0, 0.1, size=x.shape)
    return np.sum(x**2) + np.sum(noise),

# Funkcja Rastrigina z losowo zmieniającymi się parametrami
def rastrigin_dynamic(x, t):
    """Rastrigin function with dynamic parameters"""
    x = np.array(x)  # Convert input to numpy array
    A = 10 + np.sin(t)  # Dynamic change of parameter A
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)),
