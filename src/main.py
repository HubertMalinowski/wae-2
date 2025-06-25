import numpy as np
import random
from optimization import run_experiment
from functions import rosenbrock, rastrigin, quadratic
from visualization import visualize_results
import os
from datetime import datetime

# Set fixed random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def create_results_directory():
    """Create a directory for storing results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def save_results(results_df, stats, function_name, results_dir):
    """Save numerical results and statistics to files"""
    # Save final solutions
    results_file = os.path.join(results_dir, f"{function_name}_solutions.csv")
    results_df.to_csv(results_file, index=False)
    
    # Save statistics
    stats_file = os.path.join(results_dir, f"{function_name}_stats.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Statistics for {function_name}:\n")
        f.write(f"Average final fitness: {stats['avg_fitness']:.6f} ± {stats['std_fitness']:.6f}\n")
        f.write(f"Best fitness found: {stats['best_fitness']:.6f}\n")
        f.write(f"Worst fitness found: {stats['worst_fitness']:.6f}\n")
        f.write(f"Median fitness: {stats['median_fitness']:.6f}\n")

def run_all_experiments():
    # Create results directory
    results_dir = create_results_directory()
    print(f"\nResults will be saved in: {results_dir}")
    
    # Experiment parameters
    generations = 100  # number of generations
    num_runs = 30     # number of runs per experiment
    
    test_functions = [
        ("Rosenbrock", rosenbrock),
        ("Rastrigin", rastrigin),
        ("Quadratic", quadratic)
    ]
    
    for func_name, func in test_functions:
        print(f"\n{'='*50}")
        print(f"Running experiment for {func_name} function")
        print(f"Using random seed: {RANDOM_SEED}")
        print('='*50)
        
        # Create a subfolder for this function
        func_dir = os.path.join(results_dir, func_name)
        os.makedirs(func_dir, exist_ok=True)
        # Run experiment
        results, stats = run_experiment(
            func, 
            generations=generations, 
            num_runs=num_runs,
            function_name=func_name,
            save_dir=func_dir
        )
        
        # Save results
        save_results(results, stats, func_name, func_dir)
        
        # Display results
        print(f"\nFinal solutions for {func_name}:")
        print(results)
        
        # Visualize and save results
        print(f"\nVisualizing results for {func_name}...")
        plot_file = os.path.join(func_dir, f"{func_name}_visualization.png")
        visualize_results(results, title=f"Optimization Results - {func_name}", save_path=plot_file)

if __name__ == "__main__":
    run_all_experiments()
