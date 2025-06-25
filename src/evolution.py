import random
import numpy as np
from deap import base, creator, tools

# Set fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define optimization type
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Function to create individuals
# Standard DE/ES literature: search space [-5, 5] for each dimension
def create_individual():
    return [random.uniform(-5.0, 5.0) for _ in range(2)]  # 2D example

# Create and register tools in toolbox
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Differential Evolution DE/rand/1/bin with standard parameters (F=0.5, CR=0.9)
def differential_evolution(individual, population, CR=0.9, F=0.5):
    # Select 3 random individuals from the population (excluding the current individual)
    a, b, c = random.sample([ind for ind in population if ind != individual], 3)
    mutant = [a[i] + F * (b[i] - c[i]) for i in range(len(individual))]
    # Binomial crossover
    child = creator.Individual([
        mutant[i] if random.random() < CR else individual[i]
        for i in range(len(individual))
    ])
    return child

# (mu, lambda)-Evolution Strategy with standard parameters (mu=10, lambda=50)
def evolutionary_strategy(func, t, mu=10, lambda_=50, generations=1):
    population = toolbox.population(n=mu)

    # Initial evaluation
    for ind in population:
        if t is not None:  # Time-dependent function
            ind.fitness.values = func(ind, t)
        else:  # Time-independent function
            ind.fitness.values = func(ind)

    for _ in range(generations):
        offspring = []
        for _ in range(lambda_):
            parent = random.choice(population)
            child = differential_evolution(parent, population, CR=0.9, F=0.5)
            if t is not None:
                child.fitness.values = func(child, t)
            else:
                child.fitness.values = func(child)
            offspring.append(child)
        # Select the best mu individuals from lambda offspring
        population = sorted(offspring, key=lambda x: x.fitness.values[0])[:mu]
    return population
