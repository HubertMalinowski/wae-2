import numpy as np
import matplotlib.pyplot as plt

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

from deap import base, creator, tools
import random

# Ustalenie typu optymalizacji
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Funkcja do tworzenia osobników
def create_individual():
    return [random.uniform(-5.0, 5.0) for _ in range(2)]  # Przykładowo dla 2D

# Tworzenie i rejestracja narzędzi w toolbox
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Funkcja ewolucji różnicowej DE/rand/1/bin
def differential_evolution(individual, toolbox, mu=0.5, F=0.8):
    # Tworzenie populacji za pomocą toolbox
    population = list(toolbox.population(n=3))  # Upewnij się, że generujesz populację
    a, b, c = random.sample(population, 3)  # Wybór 3 losowych osobników
    mutant = [a[i] + F * (b[i] - c[i]) for i in range(len(individual))]
    
    # Utworzenie nowego osobnika jako instancji creator.Individual, aby miał atrybut fitness
    child = creator.Individual([mutant[i] if random.random() < mu else individual[i] for i in range(len(individual))])
    
    return child

# Funkcja strategii ewolucyjnej (µ, λ)
def evolutionary_strategy(mu=10, lambda_=50, generations=100):
    # Inicjalizacja populacji
    population = toolbox.population(n=mu)
    
    # Ewolucja przez µ, λ strategię
    for gen in range(generations):
        offspring = list(map(lambda ind: differential_evolution(ind, toolbox), population))
        
        # Selekcja najlepszych λ osobników
        population = sorted(offspring, key=lambda x: x.fitness.values)[:mu]
    
    return population


import pandas as pd

def run_experiment(func, generations=100, num_runs=30):
    results = []
    
    for _ in range(num_runs):
        # Losowanie początkowego stanu
        population = toolbox.population(n=10)  # 10 osobników początkowych
        best_solution = None
        
        for gen in range(generations):
            # Zmieniamy funkcję celu w zależności od niepewności
            t = np.random.rand() * 10  # Losowy czas dla zmienności
            func_value = func(np.array(population[0]), t)
            # Optymalizacja
            population = evolutionary_strategy()
            best_solution = population[0]
        
        results.append(best_solution)
    
    # Analiza wyników
    results_df = pd.DataFrame(results, columns=["x1", "x2"])
    return results_df

def visualize_results(results_df):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["x1"], results_df["x2"], 'o')
    plt.title("Wyniki optymalizacji")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

# Przykład uruchomienia eksperymentu na funkcji Rosenbrocka z dynamicznie zmieniającym się minimum
results = run_experiment(rosenbrock_dynamic)
visualize_results(results)
