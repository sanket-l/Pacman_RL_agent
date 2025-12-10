# visualize_play.py
import pickle, time, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pacman_env import PacmanEnv

log_dir = f"logs/{sys.argv[1]}_episodes"
with open(os.path.join(log_dir, f"q_table_{sys.argv[1]}.pkl"), "rb") as f:
    Q = pickle.load(f)

# load walls if saved during training
walls_path = os.path.join(log_dir, "walls.npy")
walls = np.load(walls_path) if os.path.exists(walls_path) else None
grid_size = walls.shape[0] if walls is not None else 15

env = PacmanEnv(grid_size=grid_size, fix_walls=True, walls=walls)
env.seed(0)
obs, _ = env.reset()
state = tuple(int(x) for x in obs)

frames = []
max_steps = 10000  # safeguard in case no win; can be increased
for t in range(max_steps):
    # greedy action
    if state in Q:
        action = int(np.argmax(Q[state]))
    else:
        action = env.action_space.sample()

    obs, reward, terminated, truncated, _ = env.step(action)
    img = env.render(mode="rgb_array")
    frames.append(img)
    state = tuple(int(x) for x in obs)

    if terminated or truncated:
        # win if all pellets are gone
        if np.sum(env.pellets) == 0:
            break  # stop visualization on win
        # otherwise restart episode and keep visualizing
        obs, _ = env.reset()
        state = tuple(int(x) for x in obs)

# animate with matplotlib
fig = plt.figure(figsize=(4,4))
im = plt.imshow(frames[0])
plt.axis("off")

def update(i):
    im.set_array(frames[i])
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
# show
plt.show()

# optionally save mp4 (requires ffmpeg)
save_path = os.path.join(log_dir, "playback.mp4")
ani.save(save_path, writer="ffmpeg", fps=60)
print("Saved playback to", save_path)