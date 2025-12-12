# visualize_win.py
import pickle, time, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import GRID_SIZE
from pacman_env import PacmanEnv

log_dir = f"logs/{sys.argv[1]}_episodes"
with open(os.path.join(log_dir, f"q_table_{sys.argv[1]}.pkl"), "rb") as f:
    Q = pickle.load(f)

# load walls
walls_path = os.path.join(log_dir, "walls.npy")
walls = np.load(walls_path) if os.path.exists(walls_path) else None
grid_size = walls.shape[0] if walls is not None else GRID_SIZE

env = PacmanEnv(grid_size=grid_size, walls=walls)

win_frames = None
max_search_episodes = 5000

for ep in range(max_search_episodes):
    if ep % 100 == 0:
        print(ep)
    obs, _ = env.reset()
    state = tuple(int(x) for x in obs)

    frames = []
    done = False

    while not done:
        # greedy action
        action = np.argmax(Q[state]) if state in Q else env.action_space.sample()
        
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render(mode="rgb_array")
        frames.append(frame)

        state = tuple(int(x) for x in obs)
        done = terminated or truncated

        # check win
        if done and np.sum(env.pellets) < 50:
            win_frames = frames
            print("Found a winning episode at episode", ep)
            break

    if win_frames is not None:
        break

if win_frames is None:
    print("No win found. Try increasing max_search_episodes.")
    sys.exit(0)

# Animate only the winning frames
fig = plt.figure(figsize=(4,4))
im = plt.imshow(win_frames[0])
plt.axis("off")

def update(i):
    im.set_array(win_frames[i])
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=len(win_frames), interval=60, blit=True)
plt.show()

save_path = os.path.join(log_dir, "winning_playback.mp4")
ani.save(save_path, writer="ffmpeg", fps=30)
print("Saved winning playback to", save_path)
