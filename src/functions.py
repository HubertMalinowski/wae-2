import numpy as np

# Funkcja Rosenbrocka z dynamicznie zmieniającym się minimum
def rosenbrock_dynamic(x, t):
    """Dynamicznie zmieniające się minimum funkcji Rosenbrocka"""
    a = 1.0
    b = 100.0
    x_shifted = x - np.sin(t)  # Dynamiczne przesunięcie minimum w czasie
    return (a - x_shifted[0])**2 + b * (x_shifted[1] - x_shifted[0]**2)**2

# Funkcja kwadratowa z szumem losowym
def noisy_quadratic(x):
    """Funkcja kwadratowa z losowym szumem"""
    noise = np.random.normal(0, 0.1, size=x.shape)
    return np.sum(x**2) + np.sum(noise)

# Funkcja Rastrigina z losowo zmieniającymi się parametrami
def rastrigin_dynamic(x, t):
    """Funkcja Rastrigina z dynamicznymi zmianami parametrów"""
    A = 10 + np.sin(t)  # Dynamiczna zmiana wartości A
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
