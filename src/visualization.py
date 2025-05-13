import matplotlib.pyplot as plt

def visualize_results(results_df):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["x1"], results_df["x2"], 'o')
    plt.title("Wyniki optymalizacji")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
