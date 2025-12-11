# visualize_play.py
import pickle, time, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pacman_env import PacmanEnv

if len(sys.argv) < 2:
    print("Usage: python visualize_play.py <episodes_shortname>")
    sys.exit(1)

ep_short = sys.argv[1]
log_dir = f"logs/{ep_short}_episodes"
q_path = os.path.join(log_dir, f"q_table_{ep_short}.pkl")
if not os.path.exists(q_path):
    # try alternative filename without suffix
    q_path = os.path.join(log_dir, f"q_table_{ep_short}.pkl")
if not os.path.exists(q_path):
    print("Q-table not found:", q_path)
    sys.exit(1)

with open(q_path, "rb") as f:
    Q = pickle.load(f)

# try to load walls if present
walls_path = os.path.join(log_dir, "walls.npy")
walls = np.load(walls_path) if os.path.exists(walls_path) else None
grid_size = walls.shape[0] if walls is not None else None

# create env
if grid_size is not None:
    env = PacmanEnv(grid_size=grid_size, walls=walls)
else:
    env = PacmanEnv()
env.seed(0)

frames = []
obs, _ = env.reset()
state = tuple(int(x) for x in obs)

max_steps = 2000
for t in range(max_steps):
    if state in Q:
        action = int(np.argmax(Q[state]))
    else:
        action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    img = env.render(mode="rgb_array")
    frames.append(img)
    state = tuple(int(x) for x in obs)
    if terminated or truncated:
        break

# animate
fig = plt.figure(figsize=(4,4))
im = plt.imshow(frames[0])
plt.axis("off")

def update(i):
    im.set_array(frames[i])
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=80, blit=True)
plt.show()

# optionally save mp4
save_path = os.path.join(log_dir, "playback.mp4")
try:
    ani.save(save_path, writer="ffmpeg", fps=12)
    print("Saved playback to", save_path)
except Exception as e:
    print("Could not save mp4 (ffmpeg may be missing):", e)
