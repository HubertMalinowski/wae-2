import random
import numpy as np
from deap import base, creator, tools

# Set fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
def differential_evolution(individual, population, mu=0.5, F=0.8):
    # Wybór 3 losowych osobników z populacji (bez siebie samego)
    a, b, c = random.sample([ind for ind in population if ind != individual], 3)
    mutant = [a[i] + F * (b[i] - c[i]) for i in range(len(individual))]
    
    # Krzyżowanie binarne
    child = creator.Individual([
        mutant[i] if random.random() < mu else individual[i]
        for i in range(len(individual))
    ])
    return child

# Funkcja strategii ewolucyjnej (µ, λ)
def evolutionary_strategy(func, t, mu=10, lambda_=50, generations=1):
    population = toolbox.population(n=mu)

    # Ewaluacja startowa
    for ind in population:
        if t is not None:  # Time-dependent function
            ind.fitness.values = func(ind, t)
        else:  # Time-independent function
            ind.fitness.values = func(ind)

    for _ in range(generations):
        offspring = []
        for ind in population:
            child = differential_evolution(ind, population)
            if t is not None:  # Time-dependent function
                child.fitness.values = func(child, t)
            else:  # Time-independent function
                child.fitness.values = func(child)
            offspring.append(child)
        
        # Selekcja najlepszych
        population = sorted(offspring, key=lambda x: x.fitness.values[0])[:mu]

    return population
