import random
from deap import base, creator, tools

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
