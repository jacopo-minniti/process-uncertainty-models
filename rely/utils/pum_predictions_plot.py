import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import load_dataset


data = load_dataset("jacopo-minniti/MATH-PUM-qwen2.5-1.5B", "half_entropy", split="test")
labels = [d["labels"] for d in data]

# --- Data Processing ---

# We'll look at the last 20 positions.
# You can adjust this value if you want to see more or less of the end of the sequences.
num_positions_to_check = 20

# Create a dictionary to store the counts of True/False for each position from the end.
# The structure will be: {position_from_end: {'True': count, 'False': count}}
# e.g., {-1: {'True': 100, 'False': 50}, -2: ...}
position_counts = defaultdict(lambda: defaultdict(int))

# Iterate over each list of labels
for label_list in labels:
    # Iterate from the end of the list backwards
    for i in range(1, min(len(label_list) + 1, num_positions_to_check + 1)):
        position_from_end = -i
        label_value = label_list[position_from_end]
        position_counts[position_from_end][label_value] += 1

# --- Plotting ---

# Get the positions we have data for (from -num_positions_to_check to -1)
positions = sorted(position_counts.keys())

# Extract the counts for True and False for each position
true_counts = [position_counts[pos][True] for pos in positions]
false_counts = [position_counts[pos][False] for pos in positions]

# Create the figure and axes for the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Create the stacked bar chart
bar_width = 0.8
ax.bar(positions, false_counts, bar_width, label='False (entropy <= 1.04)', color='salmon')
ax.bar(positions, true_counts, bar_width, bottom=false_counts, label='True (entropy > 1.04)', color='skyblue')

# --- Formatting the Plot ---

# Add labels, title, and legend
ax.set_xlabel("Position from the end of the list", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Frequency of True/False Labels at the End of Sequences", fontsize=14)
ax.legend()

# Set the x-axis ticks to be the positions
ax.set_xticks(positions)
ax.set_xticklabels(positions)

# Add a grid for better readability
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)

# Add text labels on the bars to show the counts
for i, pos in enumerate(positions):
    # Add text for the 'False' part of the bar
    if false_counts[i] > 0:
        ax.text(pos, false_counts[i] / 2, str(false_counts[i]), ha='center', va='center', color='white', fontweight='bold')
    # Add text for the 'True' part of the bar
    if true_counts[i] > 0:
        ax.text(pos, false_counts[i] + true_counts[i] / 2, str(true_counts[i]), ha='center', va='center', color='white', fontweight='bold')

# Make the layout tight to prevent labels from overlapping
plt.tight_layout()

# Save the plot to a file
plt.savefig("/scratch/jacopo04/gt_entropy_dist.png")

print("Plot saved as gt_entropy_dist.png")

# Also, let's print the counts for the last few positions to give a precise answer
print("\nCounts for the last 5 positions:")
for i in range(-1, -6, -1):
    if i in position_counts:
        print(f"Position {i}:")
        print(f"  False: {position_counts[i][False]}")
        print(f"  True:  {position_counts[i][True]}")
        total = position_counts[i][False] + position_counts[i][True]
        if total > 0:
            false_percentage = (position_counts[i][False] / total) * 100
            print(f"  % False: {false_percentage:.2f}%")
        print("-" * 20)