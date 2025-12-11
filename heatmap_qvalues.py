# heatmap_qvalues.py
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
from config import GRID_SIZE
from pacman_env import PacmanEnv

# -----------------------------
# Load Q-table
# -----------------------------
episodes_shortname = sys.argv[1]
log_dir = f"logs/{episodes_shortname}_episodes"

with open(os.path.join(log_dir, f"q_table_{episodes_shortname}.pkl"), "rb") as f:
    Q = pickle.load(f)

# -----------------------------
# IMPORTANT: Use same grid size as training
# -----------------------------
env = PacmanEnv(grid_size=GRID_SIZE)   # MATCH your training size

# -----------------------------
# Reconstruct training-style pellets
# -----------------------------
pellets_template = np.ones((env.grid_size, env.grid_size), dtype=int)

# pellets do not exist on walls
pellets_template[env.walls == 1] = 0

# pellet missing at start position
pellets_template[env.start_pac[0], env.start_pac[1]] = 0

pellets_flat = pellets_template.flatten().tolist()
walls_flat = env.walls.flatten().tolist()

# -----------------------------
# Build heatmap: average max-Q for each Pacman (pr,pc)
# over all ghost positions
# -----------------------------
grid = np.zeros((env.grid_size, env.grid_size), dtype=float)

for pr in range(env.grid_size):
    for pc in range(env.grid_size):

        qvals = []

        for gr in range(env.grid_size):
            for gc in range(env.grid_size):

                # full state must match training format
                state = tuple(
                    [pr, pc, gr, gc] 
                    + pellets_flat 
                    + walls_flat
                )

                if state in Q:
                    qvals.append(np.max(Q[state]))

        grid[pr, pc] = np.mean(qvals) if qvals else np.nan

# -----------------------------
# Plot heatmap
# -----------------------------
plt.figure(figsize=(6, 6))
plt.imshow(grid, origin="upper")
plt.colorbar(label="Avg max-Q")
plt.title("Heatmap: Avg max-Q per Pacman position (pellets = initial)")
plt.xlabel("Column")
plt.ylabel("Row")
plt.gca().invert_yaxis()
plt.tight_layout()

os.makedirs(log_dir, exist_ok=True)
out_path = os.path.join(log_dir, f"q_heatmap_{episodes_shortname}.png")
plt.savefig(out_path)

print(f"Saved {out_path}")
