# heatmap_qvalues.py
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
from pacman_env import PacmanEnv

episodes_shortname = sys.argv[1]
log_dir = f"logs/{episodes_shortname}_episodes"
with open(os.path.join(log_dir, f"q_table_{episodes_shortname}.pkl"), "rb") as f:
    Q = pickle.load(f)

env = PacmanEnv(grid_size=21)

# Build initial pellets template (same as reset)
pellets_template = np.ones((env.grid_size, env.grid_size), dtype=int)
pellets_template[env.start_pac[0], env.start_pac[1]] = 0
pellets_flat = pellets_template.flatten().tolist()

# For each Pacman cell, average max-Q across all ghost positions with pellets = template
grid = np.zeros((env.grid_size, env.grid_size), dtype=float)
counts = np.zeros_like(grid)

for pr in range(env.grid_size):
    for pc in range(env.grid_size):
        values = []
        for gr in range(env.grid_size):
            for gc in range(env.grid_size):
                state = tuple([pr, pc, gr, gc] + pellets_flat)
                if state in Q:
                    values.append(np.max(Q[state]))
        if len(values) > 0:
            grid[pr, pc] = np.mean(values)
            counts[pr, pc] = len(values)
        else:
            grid[pr, pc] = np.nan

plt.figure(figsize=(6,6))
plt.imshow(grid, origin="upper")
plt.colorbar(label="avg max Q")
plt.title("Heatmap: Avg max-Q per Pacman position (pellets initial)")
plt.xlabel("col")
plt.ylabel("row")
plt.gca().invert_yaxis()
plt.tight_layout()
os.makedirs(log_dir, exist_ok=True)
plt.savefig(os.path.join(log_dir, f"q_heatmap_{episodes_shortname}.png"))
print(f"Saved q_heatmap_{episodes_shortname}.png to", log_dir)