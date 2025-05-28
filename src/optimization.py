import numpy as np
import pandas as pd
from evolution import evolutionary_strategy, toolbox
from functions import rosenbrock_dynamic
import matplotlib.pyplot as plt
from deap import creator
import inspect

def run_experiment(func, generations=100, num_runs=30, function_name="Test Function"):
    results = []
    all_fitness_history = []
    all_final_fitness = []
    
    print(f"\nStarting optimization for {function_name}")
    print(f"Parameters: generations={generations}, num_runs={num_runs}")
    
    # Check if function accepts time parameter
    is_dynamic = len(inspect.signature(func).parameters) > 1
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        population = toolbox.population(n=10)
        
        # Evaluate initial population
        t0 = 0  # Start from t=0
        for ind in population:
            if is_dynamic:
                ind.fitness.values = func(ind, t0)
            else:
                ind.fitness.values = func(ind)
        
        best_solution = min(population, key=lambda ind: ind.fitness.values[0])
        best_fitness = best_solution.fitness.values[0]
        run_fitness_history = [best_fitness]
        
        print(f"Initial best fitness: {best_fitness:.6f}")
        
        for gen in range(generations):
            t = (gen / generations) * 10 if is_dynamic else 0
            
            # For time-independent functions, we don't pass the time parameter
            if is_dynamic:
                population = evolutionary_strategy(func, t, mu=10, lambda_=50, generations=1)
            else:
                population = evolutionary_strategy(func, None, mu=10, lambda_=50, generations=1)
            
            current_best = min(population, key=lambda ind: ind.fitness.values[0])
            if current_best.fitness.values[0] < best_fitness:
                best_solution = creator.Individual(current_best)
                best_fitness = current_best.fitness.values[0]
                print(f"Generation {gen}: New best fitness = {best_fitness:.6f}")
            
            run_fitness_history.append(best_fitness)

        print(f"Final best solution: x1={best_solution[0]:.6f}, x2={best_solution[1]:.6f}")
        print(f"Final best fitness: {best_fitness:.6f}")
        
        results.append(best_solution)
        all_fitness_history.append(run_fitness_history)
        all_final_fitness.append(best_fitness)
    
    # Calculate statistics across all runs
    avg_final_fitness = np.mean(all_final_fitness)
    std_final_fitness = np.std(all_final_fitness)
    best_final_fitness = min(all_final_fitness)
    worst_final_fitness = max(all_final_fitness)
    median_final_fitness = np.median(all_final_fitness)
    
    stats = {
        'avg_fitness': avg_final_fitness,
        'std_fitness': std_final_fitness,
        'best_fitness': best_final_fitness,
        'worst_fitness': worst_final_fitness,
        'median_fitness': median_final_fitness
    }
    
    print(f"\nOverall Statistics for {function_name}:")
    print(f"Average final fitness: {avg_final_fitness:.6f} ± {std_final_fitness:.6f}")
    print(f"Best fitness found: {best_final_fitness:.6f}")
    print(f"Worst fitness found: {worst_final_fitness:.6f}")
    print(f"Median fitness: {median_final_fitness:.6f}")
    
    # Plot convergence for all runs
    plt.figure(figsize=(12, 6))
    for i, hist in enumerate(all_fitness_history):
        plt.plot(hist, alpha=0.2, color='blue')
    
    # Plot average convergence
    avg_convergence = np.mean(all_fitness_history, axis=0)
    plt.plot(avg_convergence, color='red', linewidth=2, label='Average across runs')
    
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"Convergence Plot - {function_name}")
    plt.yscale('log')  # Using log scale to better see improvements
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    results_df = pd.DataFrame([[ind[0], ind[1]] for ind in results], columns=["x1", "x2"])
    return results_df, stats
