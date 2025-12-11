# train_q_learning.py
import numpy as np
import pickle, csv, os, sys
from pacman_env import PacmanEnv
from collections import defaultdict
import matplotlib.pyplot as plt

env = PacmanEnv(grid_size=21)
env.seed(0)

def obs_to_state(obs):
    # convert numpy obs vector to an int tuple for dict key
    return tuple(int(x) for x in obs)

Q = defaultdict(lambda: np.zeros(env.action_space.n))

# hyperparams
episodes_shortname = sys.argv[1]
episodes = int(episodes_shortname)
max_steps = 700
alpha = 0.03             # stable learning rate
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.02
epsilon_decay = 0.05

log_dir = "logs/" + episodes_shortname + "_episodes"
os.makedirs(log_dir, exist_ok=True)

metrics = {"episode":[], "reward":[], "length":[], "epsilon":[], "palletsEaten":[]}

for ep in range(1, episodes + 1):
    obs, _ = env.reset()
    state = obs_to_state(obs)
    total_reward = 0
    length = 0
    palletsEaten = 0

    for t in range(max_steps):
        # epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        obs, reward, terminated, truncated, _ = env.step(action)

        if reward > 0:
            palletsEaten += 1
        done = terminated or truncated
        next_state = obs_to_state(obs)

        # Q update
        best_next = np.max(Q[next_state])
        # Correct terminal handling
        if done:
            td_target = reward
        else:
            td_target = reward + gamma * best_next

        Q[state][action] += alpha * (td_target - Q[state][action])

        state = next_state
        total_reward += reward
        length += 1
        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    metrics["episode"].append(ep)
    metrics["reward"].append(total_reward)
    metrics["palletsEaten"].append(palletsEaten)
    metrics["length"].append(length)
    metrics["epsilon"].append(epsilon)

    if ep % 100 == 0 or ep == 1:
        print(f"Ep {ep}/{episodes} reward={total_reward:.1f} len={length} eps={epsilon:.3f}")

# save Q-table
with open(os.path.join(log_dir, f"q_table_{episodes_shortname}.pkl"), "wb") as f:
    pickle.dump(dict(Q), f)
# save walls used during training (only meaningful if fix_walls=True)
walls_path = os.path.join(log_dir, "walls.npy")
if getattr(env, "_fixed_walls_cache", None) is not None:
    np.save(walls_path, env._fixed_walls_cache)
    print("Saved walls to", walls_path)

# save metrics CSV
csv_path = os.path.join(log_dir, f"metrics_{episodes_shortname}.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "length", "epsilon"])
    for i in range(len(metrics["episode"])):
        writer.writerow([metrics["episode"][i], metrics["reward"][i], metrics["length"][i], metrics["epsilon"][i]])

# Moving average smoothing
window = 200  # you can try 100, 200, 300
pallets = np.array(metrics["palletsEaten"])
smoothed = np.convolve(pallets, np.ones(window)/window, mode="valid")

plt.figure(figsize=(8,4))

# raw curve (very noisy)
plt.plot(metrics["episode"], metrics["palletsEaten"], alpha=0.3, label="Raw")

# smoothed curve
plt.plot(metrics["episode"][window-1:], smoothed, linewidth=2, label=f"Smoothed (window={window})")

plt.xlabel("Episode")
plt.ylabel("Pallets eaten")
plt.title("Learning curve (Q-learning)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(log_dir, f"learning_curve_{episodes_shortname}.png"))


# # plot learning curve
# plt.figure(figsize=(8,4))
# plt.plot(metrics["episode"], metrics["palletsEaten"], label="Episode reward")
# plt.xlabel("Episode")
# plt.ylabel("Pallets eaten")
# plt.title("Learning curve (Q-learning)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(log_dir, f"learning_curve_{episodes_shortname}.png"))
# print(f"Saved learning_curve_{episodes_shortname}.png and q_table_{episodes_shortname}.pkl in", log_dir)