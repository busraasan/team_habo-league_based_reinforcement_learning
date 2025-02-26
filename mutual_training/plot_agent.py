import matplotlib.pyplot as plt
import numpy as np

# Load the data from the text file
file_path = "results/td3_selfplay_add_sac_5M/win_rates_sac.txt"  # Change this to your actual file path
file_path_strong = "results/td3_selfplay_add_sac_5M/win_rates.txt"  # Change this to your actual file path

file_path_n = "results/td3_selfplay_5M_model/win_rates_sac.txt"  # Change this to your actual file path
file_path_strong_n = "results/td3_selfplay_5M_model/win_rates.txt"  # Change this to your actual file path

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    step_sizes = []
    win_rates = []
    for line in lines:
        parts = line.strip().split(',')
        step_size = float(parts[0])
        win_rate = float(parts[1].strip('[]'))  # Extract the first win rate value
        
        step_sizes.append(step_size)
        win_rates.append(win_rate)
    
    return np.array(step_sizes), np.array(win_rates)

# Load data

step_sizes, win_rates_sac_normal = load_data(file_path_n)
step_sizes, win_rates_normal = load_data(file_path_strong_n)

step_sizes, win_rates_sac = load_data(file_path)
step_sizes, win_rates = load_data(file_path_strong)


# Plot the data
plt.figure(figsize=(8, 5))

# Plot the first line (SAC win rate)
plt.plot(step_sizes, win_rates_sac, color='#8B008B', linewidth=2, linestyle='-', label="vs SAC - w SAC")  # Dark Magenta
plt.plot(step_sizes[:41], win_rates_sac_normal[:41], color='#FF69B4', linewidth=2, linestyle='-', label="vs SAC")  # Light Magenta
plt.plot(step_sizes, win_rates, color='#008080', linewidth=2, linestyle='-', label="vs Strong Opponent - w SAC")  # Teal
plt.plot(step_sizes[:41], win_rates_normal[:41], color='#90EE90', linewidth=2, linestyle='-', label="vs Strong Opponent")  # Light Gree

# Labels and title
plt.xlabel("Step Size", fontweight='bold', fontsize=14)
plt.ylabel("Win Rate", fontweight='bold', fontsize=14)
plt.title("Win Rate per Step Size", fontweight='bold', fontsize=16)

# Adjust x-axis labels (if step sizes are in millions)
plt.yticks(fontsize=12)

# Grid and legend
plt.grid()
# put legend in upper left
plt.legend(loc='lower right', fontsize=12)  # This will show labels for the lines

plt.savefig('win_rates_graph.png', dpi=300, bbox_inches='tight')

