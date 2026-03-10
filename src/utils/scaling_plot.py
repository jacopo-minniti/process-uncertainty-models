import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import json
import sys
import argparse


################# Configuration #################
if len(sys.argv) > 1:
    STRATEGY = sys.argv[1]
else:
    STRATEGY = 'bon'
print(f"Using strategy: {STRATEGY}")

SHOW_ALL_POINTS = True  # Set to True to show all points, not just Pareto frontier
DATA_FILE_PATH = "results/results_mine.json"
SHOW_MAJ_VOTING = False

# "Maj@32": .5907,
majority_voting = {
    "Maj@4": .482,
    "Maj@8": .52,
    "Maj@100": .5932,
}

# --- 1. Helper Function to Calculate FLOPs ---
def flops(tokens):
    """
    Calculates the flops based on the number of tokens.
    The result is returned as a power of 2 (log2).
    """
    # Estimate FLOPs based on a simplified model
    # 2.62B is an example parameter, representing FLOPs per token
    flops_val = 2_620_000_000 * tokens
    # Convert to log base 2 for the plot's x-axis
    return round(np.log2(flops_val), 2)

def is_pareto_dominant(data):
    """
    Identifies Pareto-dominant points based on accuracy (higher is better) 
    and computational cost (lower is better).
    Returns a list of indices of Pareto-dominant points.
    """
    pareto_indices = []
    
    for i, point_i in enumerate(data):
        flops_i = flops(point_i['tokens generated'])
        accuracy_i = point_i[STRATEGY]
        
        is_dominated = False
        for j, point_j in enumerate(data):
            if i == j:
                continue
                
            flops_j = flops(point_j['tokens generated'])
            accuracy_j = point_j[STRATEGY]
            
            # Point i is dominated if point j has better or equal accuracy AND lower or equal FLOPs
            # and at least one of these is strictly better
            if (accuracy_j >= accuracy_i and flops_j <= flops_i and 
                (accuracy_j > accuracy_i or flops_j < flops_i)):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_indices.append(i)
    
    return pareto_indices


# --- 2. Data Input ---
# Load algorithm data from JSON file
with open(DATA_FILE_PATH, "r") as f:
    algorithms_data = json.load(f)["data"]

# --- 3. Plotting Configuration ---
# Generate distinct colors and markers for each algorithm
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

# --- 4. Create the Plot ---
fig, ax = plt.subplots(figsize=(12, 8))

# Plot all algorithms
for i, algorithm in enumerate(algorithms_data):
    name = algorithm["name"]
    data = algorithm["data"]
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    
    # Calculate Pareto-dominant points
    pareto_indices = is_pareto_dominant(data)
    pareto_points = [(flops(data[j]['tokens generated']), data[j][STRATEGY]) for j in pareto_indices]
    pareto_points.sort()
    
    # Plot all points if SHOW_ALL_POINTS is True
    if SHOW_ALL_POINTS:
        all_x = [flops(point['tokens generated']) for point in data]
        all_y = [point[STRATEGY] for point in data]
        ax.scatter(all_x, all_y, color=color, marker=marker, s=50,
                   edgecolors='black', alpha=0.5, zorder=1)
    
    # Plot Pareto points
    if pareto_points:
        pareto_x, pareto_y = zip(*pareto_points)
        ax.scatter(pareto_x, pareto_y, color=color, marker=marker, s=100,
                   edgecolors='black', alpha=0.8, label=name, zorder=3)
        ax.plot(pareto_x, pareto_y, color=color, linestyle='--', linewidth=2, alpha=0.7)

# Add horizontal lines for majority voting baselines
if SHOW_MAJ_VOTING:
    for name, accuracy in majority_voting.items():
        ax.axhline(y=accuracy, color='gray', linestyle='--', alpha=0.6, zorder=0)
        # Add text label for the line
        ax.text(ax.get_xlim()[1], accuracy, f' {name}', verticalalignment='center', fontsize=9, color='gray')

# --- 5. Customize the Plot ---
ax.set_title('Algorithm Comparison: Performance vs Computational Cost', fontsize=16, pad=20)
ax.set_xlabel('Inference FLOPs ($log_2$)', fontsize=12)
ax.set_ylabel(f"Accuracy ({STRATEGY})", fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
ax.legend(fontsize=11, loc='upper left')

# Adjust layout and save
plt.tight_layout()
plt.savefig(f"assets/figures/sbs-pareto-{STRATEGY}.png", dpi=300, bbox_inches='tight')
plt.show()

# --- 6. Print Pareto Analysis ---
print("=== Pareto Analysis ===")
for algorithm in algorithms_data:
    name = algorithm["name"]
    data = algorithm["data"]
    pareto_indices = is_pareto_dominant(data)
    print(f"{name}: {len(pareto_indices)} Pareto-dominant points out of {len(data)} total")
    
    # Print Pareto points sorted by FLOPs
    pareto_points = [(i, data[i]) for i in pareto_indices]
    pareto_points.sort(key=lambda x: flops(x[1]['tokens generated']))
    
    for _, point in pareto_points:
        print(f"  B1={point['B1']}, B3={point['B3']}, "
              f"FLOPs(log2)={flops(point['tokens generated']):.1f}, "
              f"Accuracy={point[STRATEGY]:.4f}")