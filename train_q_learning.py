# train_q_learning.py
import numpy as np
import pickle, csv, os, sys
import random
from pacman_env import PacmanEnv
from collections import defaultdict
import matplotlib.pyplot as plt

env = PacmanEnv(grid_size=15, fix_walls=True)
env.seed(0)
np.random.seed(0)
random.seed(0)

def obs_to_state(obs):
    # convert numpy obs vector to an int tuple for dict key
    return tuple(int(x) for x in obs)

Q = defaultdict(lambda: np.zeros(env.action_space.n))

# visitation counts for novelty-based intrinsic reward
visited_counts = defaultdict(int)

# reward-shaping hyperparams
coin_bonus = 0.5        # extra reward when a coin is collected
empty_penalty = -0.01   # small penalty for stepping on empty cells
novelty_scale = 0.2     # scale for novelty (intrinsic) bonus

# hyperparams
episodes_shortname = sys.argv[1] if len(sys.argv) > 1 else "100"
episodes = int(episodes_shortname)
alpha = 0.1
# alpha (learning rate) schedule: multiplicative decay to alpha_min over
# roughly half the episodes to stabilize updates later in training
alpha_init = alpha
alpha_min = 0.01
target_alpha_decay_episodes = max(1, episodes // 2)
alpha_decay = (alpha_min / alpha_init) ** (1.0 / target_alpha_decay_episodes)
gamma = 0.99
# epsilon schedule: start -> min; compute multiplicative decay so epsilon_min
# is reached after roughly half of the total episodes (faster decay)
epsilon_init = 1.0
epsilon_min = 0.05
target_decay_episodes = max(1, episodes // 2)
epsilon_decay = (epsilon_min / epsilon_init) ** (1.0 / target_decay_episodes)
epsilon = epsilon_init
max_steps = 10000

log_dir = "logs/" + episodes_shortname + "_episodes"
os.makedirs(log_dir, exist_ok=True)

metrics = {"episode":[], "reward":[], "shaped_reward":[], "length":[], "epsilon":[]}

for ep in range(1, episodes + 1):
    obs, _ = env.reset()
    state = obs_to_state(obs)
    total_reward = 0
    shaped_total = 0
    length = 0

    # count the starting state as visited
    visited_counts[state] += 1

    for t in range(max_steps):
        # epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = obs_to_state(obs)
        # reward shaping: coin bonus / empty penalty + novelty bonus
        novelty_bonus = novelty_scale / np.sqrt(visited_counts[next_state] + 1)
        # clip per-step novelty to avoid large spikes
        novelty_cap = 0.3
        novelty_bonus = min(novelty_bonus, novelty_cap)
        if reward > 0:
            shaped_reward = reward + coin_bonus + novelty_bonus
        else:
            shaped_reward = reward + empty_penalty + novelty_bonus
        # clip shaped reward per step to a stable range for logging
        shaped_reward = float(np.clip(shaped_reward, -1.0, 2.0))

        # Q update uses only the environment reward (no intrinsic/coin/empty bonuses)
        # This keeps the learning target stationary while still tracking shaped
        # reward separately for logging/analysis.
        best_next = np.max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

        # advance
        state = next_state
        visited_counts[next_state] += 1
        total_reward += reward
        shaped_total += shaped_reward
        length += 1
        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    # decay alpha per episode (keep within bounds)
    alpha = max(alpha_min, alpha * alpha_decay)

    metrics["episode"].append(ep)
    metrics["reward"].append(total_reward)
    metrics["shaped_reward"].append(shaped_total)
    metrics["length"].append(length)
    metrics["epsilon"].append(epsilon)

    if ep % 100 == 0 or ep == 1:
        print(f"Ep {ep}/{episodes} reward={total_reward:.1f} shaped={shaped_total:.1f} len={length} eps={epsilon:.3f}")

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
    writer.writerow(["episode", "reward", "shaped_reward", "length", "epsilon"])
    for i in range(len(metrics["episode"])):
        writer.writerow([metrics["episode"][i], metrics["reward"][i], metrics["shaped_reward"][i], metrics["length"][i], metrics["epsilon"][i]])

# plot learning curve (raw and smoothed)
plt.figure(figsize=(8,4))
plt.plot(metrics["episode"], metrics["shaped_reward"], alpha=0.4, label="Shaped episode reward (raw)")

# moving-average smoothing window (use 50 or smaller if fewer episodes)
window = 50 if episodes >= 50 else max(1, episodes // 2)
if len(metrics["shaped_reward"]) >= window:
    weights = np.ones(window) / window
    smoothed = np.convolve(metrics["shaped_reward"], weights, mode='valid')
    x_smoothed = metrics["episode"][window - 1:]
    plt.plot(x_smoothed, smoothed, color='tab:orange', label=f"Smoothed (window={window})")

plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Learning curve (Q-learning)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(log_dir, f"learning_curve_{episodes_shortname}.png"))
print(f"Saved learning_curve_{episodes_shortname}.png and q_table_{episodes_shortname}.pkl in", log_dir)