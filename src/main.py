from optimization import run_experiment
from functions import rosenbrock_dynamic
from visualization import visualize_results

def main():
    # Parametry eksperymentu
    generations = 100  # liczba pokoleń
    num_runs = 30      # liczba uruchomień eksperymentu

    # Uruchomienie eksperymentu
    print("Uruchamianie eksperymentu...")
    results = run_experiment(rosenbrock_dynamic, generations=generations, num_runs=num_runs)
    
    # Wyświetlenie wyników
    print("Wyniki eksperymentu:")
    print(results)
    
    # Wizualizacja wyników
    print("Wizualizacja wyników...")
    visualize_results(results)

if __name__ == "__main__":
    main()
