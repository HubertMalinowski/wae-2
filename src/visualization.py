import matplotlib.pyplot as plt

def visualize_results(results_df, title="Optimization Results", save_path=None):
    """
    Visualize optimization results and optionally save to file
    
    Args:
        results_df: DataFrame with x1 and x2 coordinates
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["x1"], results_df["x2"], 'o', alpha=0.6)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
