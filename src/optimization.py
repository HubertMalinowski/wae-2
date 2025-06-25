import numpy as np
import pandas as pd
from evolution import evolutionary_strategy, toolbox
from functions import rosenbrock
import matplotlib.pyplot as plt
from deap import creator
import inspect
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

# Helper: get optimum for each function at t=0
optima = {
    'Rosenbrock': (1 + np.sin(0), 1 + np.sin(0)),
    'Quadratic': (np.sin(0), np.sin(0)),
    'Rastrigin': (0, 0)
}

def save_function_landscape(func, function_name, save_dir):
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_vec = np.array([X[i, j], Y[i, j]])
            z_val = func(x_vec, 0)
            if isinstance(z_val, tuple):
                z_val = z_val[0]
            Z[i, j] = z_val
    # 3D surface with optimum
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(121, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title(f'Surface plot of {function_name} at t=0')
    fig.colorbar(surf, shrink=0.5, aspect=5, ax=ax)
    # Optimum marker
    if function_name in optima:
        opt = optima[function_name]
        ax.scatter(opt[0], opt[1], func(opt, 0)[0], color='gold', s=100, marker='*', label='Optimum')
        ax.legend()
    # 2D contour
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis')
    fig.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Contour at t=0')
    if function_name in optima:
        ax2.scatter(*optima[function_name], color='gold', s=100, marker='*', label='Optimum')
        ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{function_name}_landscape.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"{function_name}_landscape.svg"))
    plt.close()
    return X, Y, Z

def save_convergence_plot(all_fitness_history, function_name, all_final_fitness, save_dir):
    mean = np.mean(all_fitness_history, axis=0)
    std = np.std(all_fitness_history, axis=0)
    median = np.median(all_fitness_history, axis=0)
    plt.figure(figsize=(10, 6))
    for hist in all_fitness_history:
        plt.plot(hist, color='gray', alpha=0.15)
    plt.plot(mean, color='C0', label='Mean')
    plt.plot(median, color='C1', label='Median')
    plt.fill_between(range(len(mean)), mean-std, mean+std, color='C0', alpha=0.2, label='±1 std')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.yscale('log')
    plt.title(f'Convergence Across {len(all_fitness_history)} Runs')
    # Annotate best/worst/median final fitness
    # plt.annotate(f"Best: {np.min(all_final_fitness):.2e}", xy=(1, np.min(all_final_fitness)), xycoords=('axes fraction', 'data'), color='green')
    # plt.annotate(f"Worst: {np.max(all_final_fitness):.2e}", xy=(1, np.max(all_final_fitness)), xycoords=('axes fraction', 'data'), color='red')
    # plt.annotate(f"Median: {np.median(all_final_fitness):.2e}", xy=(1, np.median(all_final_fitness)), xycoords=('axes fraction', 'data'), color='C1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{function_name}_all_runs_convergence.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"{function_name}_all_runs_convergence.svg"))
    plt.close()

def save_population_3d_plot(X, Y, Z, initial_pop, final_pop, func, function_name, run_idx, save_dir):
    # 3D plot
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    initial_z = [func(np.array(ind), 0)[0] for ind in initial_pop]
    final_z = [func(np.array(ind), 0)[0] for ind in final_pop]
    ax.scatter([ind[0] for ind in initial_pop], [ind[1] for ind in initial_pop], initial_z, color='red', s=50, label='Initial')
    sc = ax.scatter([ind[0] for ind in final_pop], [ind[1] for ind in final_pop], final_z, c=final_z, cmap='coolwarm', s=50, label='Final')
    if function_name in optima:
        opt = optima[function_name]
        ax.scatter(opt[0], opt[1], func(opt, 0)[0], color='gold', s=100, marker='*', label='Optimum')
    ax.set_title(f'3D Populations - {function_name} Run {run_idx+1}')
    ax.legend()
    # 2D projection
    ax2 = fig.add_subplot(122)
    ax2.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.5)
    ax2.scatter([ind[0] for ind in initial_pop], [ind[1] for ind in initial_pop], color='red', s=50, label='Initial')
    sc2 = ax2.scatter([ind[0] for ind in final_pop], [ind[1] for ind in final_pop], c=final_z, cmap='coolwarm', s=50, label='Final')
    if function_name in optima:
        ax2.scatter(*optima[function_name], color='gold', s=100, marker='*', label='Optimum')
    plt.colorbar(sc2, ax=ax2, label='Final Fitness')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('2D Projection')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{function_name}_run_{run_idx+1:02d}_populations.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"{function_name}_run_{run_idx+1:02d}_populations.svg"))
    plt.close()

def save_final_solutions_scatter(results, function_name, func, save_dir):
    results_df = pd.DataFrame([[ind[0], ind[1]] for ind in results], columns=['x1', 'x2'])
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_vec = np.array([X[i, j], Y[i, j]])
            z_val = func(x_vec, 0)
            if isinstance(z_val, tuple):
                z_val = z_val[0]
            Z[i, j] = z_val
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.5)
    # Color by fitness
    fitnesses = [func(np.array([row[0], row[1]]), 0)[0] for row in results_df.values]
    sc = plt.scatter(results_df['x1'], results_df['x2'], c=fitnesses, cmap='coolwarm', s=50, edgecolor='k', label='Final Solutions')
    # Add run numbers to each point
    for i, (x, y) in enumerate(zip(results_df['x1'], results_df['x2'])):
        plt.text(x, y, str(i+1), fontsize=9, ha='center', va='center', color='black', weight='bold')
    # Optimum marker (optional)
    if function_name in optima:
        plt.scatter(*optima[function_name], color='gold', s=100, marker='*', label='Optimum')
    plt.colorbar(sc, label='Fitness')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Final Solutions - {function_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{function_name}_final_solutions.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"{function_name}_final_solutions.svg"))
    plt.close()

def save_final_fitness_boxplot(all_final_fitness, function_name, save_dir):
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=all_final_fitness, inner=None, color='lightblue')
    sns.swarmplot(y=all_final_fitness, color='black', size=3)
    plt.title(f'Final Fitness Distribution - {function_name}')
    plt.ylabel('Final Fitness')
    # Annotate
    # plt.annotate(f"Mean: {np.mean(all_final_fitness):.2e}", xy=(0, np.mean(all_final_fitness)), color='C0')
    # plt.annotate(f"Median: {np.median(all_final_fitness):.2e}", xy=(0, np.median(all_final_fitness)), color='C1')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{function_name}_final_fitness_boxplot.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"{function_name}_final_fitness_boxplot.svg"))
    plt.close()

def run_experiment(func, generations=100, num_runs=30, function_name="Test Function", save_dir="."):
    results = []
    all_fitness_history = []
    all_final_fitness = []
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nStarting optimization for {function_name}")
    print(f"Parameters: generations={generations}, num_runs={num_runs}")
    is_dynamic = len(inspect.signature(func).parameters) > 1
    # Save function landscape ONCE per function
    X, Y, Z = save_function_landscape(func, function_name, save_dir)
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        population = toolbox.population(n=10)
        initial_population = [ind[:] for ind in population]
        t0 = 0  # Start from t=0
        for ind in population:
            if is_dynamic:
                ind.fitness.values = func(ind, t0)
            else:
                ind.fitness.values = func(ind)
        best_solution = min(population, key=lambda ind: ind.fitness.values[0])
        best_fitness = best_solution.fitness.values[0]
        run_fitness_history = [best_fitness]
        for gen in range(generations):
            t = (gen / generations) * 10 if is_dynamic else 0
            if is_dynamic:
                population = evolutionary_strategy(func, t, mu=10, lambda_=50, generations=1)
            else:
                population = evolutionary_strategy(func, None, mu=10, lambda_=50, generations=1)
            current_best = min(population, key=lambda ind: ind.fitness.values[0])
            if current_best.fitness.values[0] < best_fitness:
                best_solution = creator.Individual(current_best)
                best_fitness = current_best.fitness.values[0]
            run_fitness_history.append(best_fitness)
        results.append(best_solution)
        all_fitness_history.append(run_fitness_history)
        all_final_fitness.append(best_fitness)
        # Save per-run population movement plots
        save_population_3d_plot(
            X, Y, Z,
            initial_population, population,
            func, function_name, run, save_dir
        )
    # Save summary plots after all runs
    save_convergence_plot(all_fitness_history, function_name, all_final_fitness, save_dir)
    save_final_solutions_scatter(results, function_name, func, save_dir)
    save_final_fitness_boxplot(all_final_fitness, function_name, save_dir)
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
    results_df = pd.DataFrame([[ind[0], ind[1]] for ind in results], columns=["x1", "x2"])
    return results_df, stats
