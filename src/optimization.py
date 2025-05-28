import numpy as np
import pandas as pd
from evolution import evolutionary_strategy, toolbox
from functions import rosenbrock_dynamic

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
