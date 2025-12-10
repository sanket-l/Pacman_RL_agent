# visualize_play.py
import pickle, time, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pacman_env import PacmanEnv

log_dir = "logs"
with open(os.path.join(log_dir, "q_table.pkl"), "rb") as f:
    Q = pickle.load(f)

env = PacmanEnv(grid_size=7)
obs, _ = env.reset()
state = tuple(int(x) for x in obs)

frames = []
max_steps = 300
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
        break

# animate with matplotlib
fig = plt.figure(figsize=(4,4))
im = plt.imshow(frames[0])
plt.axis("off")

def update(i):
    im.set_array(frames[i])
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True)
# show
plt.show()

# optionally save mp4 (requires ffmpeg)
save_path = os.path.join(log_dir, "playback.mp4")
ani.save(save_path, writer="ffmpeg", fps=4)
print("Saved playback to", save_path)
