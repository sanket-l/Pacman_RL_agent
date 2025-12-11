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
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995
max_steps = 300

log_dir = "logs/" + episodes_shortname + "_episodes"
os.makedirs(log_dir, exist_ok=True)

metrics = {"episode":[], "reward":[], "length":[], "epsilon":[]}

for ep in range(1, episodes + 1):
    obs, _ = env.reset()
    state = obs_to_state(obs)
    total_reward = 0
    length = 0

    for t in range(max_steps):
        # epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = obs_to_state(obs)

        # Q update
        best_next = np.max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

        state = next_state
        total_reward += reward
        length += 1
        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    metrics["episode"].append(ep)
    metrics["reward"].append(total_reward)
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

# plot learning curve
plt.figure(figsize=(8,4))
plt.plot(metrics["episode"], metrics["reward"], label="Episode reward")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Learning curve (Q-learning)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, f"learning_curve_{episodes_shortname}.png"))
print(f"Saved learning_curve_{episodes_shortname}.png and q_table_{episodes_shortname}.pkl in", log_dir)