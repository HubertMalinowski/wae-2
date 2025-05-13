import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

# Funkcje testowe z niepewnością

# Funkcja kwadratowa z losowym szumem
def noisy_quadratic(x, noise_level=0.1):
    noise = np.random.normal(0, noise_level)
    return np.sum(np.square(x)) + noise

# Funkcja Rosenbrocka z dynamicznie zmieniającym się minimum
def dynamic_rosenbrock(x, t=0):
    shift = 0.1 * np.sin(0.1 * t)
    x_shifted = x + shift
    return sum(100.0 * (x_shifted[1:] - x_shifted[:-1]**2.0)**2.0 + (1 - x_shifted[:-1])**2.0)

# Funkcja Rastrigina z losowo zmieniającymi się parametrami
def rastrigin(x, noise_scale=0.1):
    A = 10
    noise = np.random.uniform(-noise_scale, noise_scale, size=len(x))
    x_noisy = x + noise
    return A * len(x) + sum(x_noisy**2 - A * np.cos(2 * np.pi * x_noisy))
