# train_q_learning_compact.py
import numpy as np
from collections import defaultdict, deque
import pickle, csv, os, sys
from pacman_env import PacmanEnv
import matplotlib.pyplot as plt

# ---------------------------
# Compact observation -> discrete state mapping
# ---------------------------
def sign(x):
    return 0 if x < 0 else (2 if x > 0 else 1)  # map negative->0, zero->1, positive->2

def dist_bucket(d):
    # bucket Manhattan distance into 0=near (<=2),1=mid(3..5),2=far(>5)
    if d <= 2:
        return 0
    elif d <= 5:
        return 1
    else:
        return 2

def obs_to_state(env, raw_obs):
    """
    Convert the raw observation returned by env._get_obs into a small discrete tuple:
    (pac_r, pac_c, pellet_dx_sign, pellet_dy_sign, ghost_dx_sign, ghost_dy_sign, ghost_dist_bucket)
    - pellet direction: direction to nearest pellet (signs -1/0/+1 mapped to 0/1/2)
    - ghost direction: direction to nearest ghost (same sign mapping)
    - ghost distance bucket: 0/1/2
    """
    s = raw_obs
    grid = env.grid_size
    pac_r, pac_c = int(s[0]), int(s[1])
    # ghosts start at index 2
    ghosts_flat = s[2:2 + 2 * env.num_ghosts].astype(int)
    ghosts = ghosts_flat.reshape((env.num_ghosts, 2))
    pellets_flat = s[2 + 2 * env.num_ghosts : 2 + 2 * env.num_ghosts + grid * grid].astype(int)
    pellets_grid = pellets_flat.reshape((grid, grid))

    # find nearest pellet (manhattan)
    pellet_positions = np.argwhere(pellets_grid == 1)
    if pellet_positions.shape[0] == 0:
        pellet_dx_sign = 1  # zero direction
        pellet_dy_sign = 1
    else:
        dists = np.abs(pellet_positions[:,0] - pac_r) + np.abs(pellet_positions[:,1] - pac_c)
        idx = np.argmin(dists)
        pr, pc = pellet_positions[idx]
        pellet_dx_sign = sign(pr - pac_r)
        pellet_dy_sign = sign(pc - pac_c)

    # nearest ghost
    g_dists = [abs(int(g[0]) - pac_r) + abs(int(g[1]) - pac_c) for g in ghosts]
    nearest_idx = int(np.argmin(g_dists))
    gr, gc = int(ghosts[nearest_idx,0]), int(ghosts[nearest_idx,1])
    ghost_dx_sign = sign(gr - pac_r)
    ghost_dy_sign = sign(gc - pac_c)
    ghost_dist = g_dists[nearest_idx]
    g_bucket = dist_bucket(ghost_dist)

    # pack into tuple (small cardinality)
    return (pac_r, pac_c, pellet_dx_sign, pellet_dy_sign, ghost_dx_sign, ghost_dy_sign, g_bucket)

# ---------------------------
# Q-learning training
# ---------------------------
def train(episodes=2000, grid_size=11, num_ghosts=2, out_dir="logs_compact"):
    env = PacmanEnv(grid_size=grid_size, num_ghosts=num_ghosts)
    env.seed(0)
    Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float32))

    # hyperparams
    alpha = 0.2
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9992
    max_steps = 500

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"metrics_{episodes}.csv")
    plot_path = os.path.join(out_dir, f"learning_curve_{episodes}.png")
    qtable_path = os.path.join(out_dir, f"q_table_{episodes}.pkl")

    metrics = {"episode": [], "reward": [], "length": [], "epsilon": [], "eval_mean_5": []}
    eval_queue = deque(maxlen=5)

    for ep in range(1, episodes + 1):
        raw_obs, _ = env.reset()
        state = obs_to_state(env, raw_obs)
        total_reward = 0.0
        length = 0

        for t in range(max_steps):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            raw_next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = obs_to_state(env, raw_next_obs)

            # Q update (tabular)
            best_next = np.max(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

            state = next_state
            total_reward += reward
            length += 1
            if terminated or truncated:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        metrics["episode"].append(ep)
        metrics["reward"].append(total_reward)
        metrics["length"].append(length)
        metrics["epsilon"].append(epsilon)

        # periodic quick evaluation (greedy policy averaged over 1 episode)
        if ep % 50 == 0:
            eval_reward = evaluate_policy(env, Q, obs_to_state, greedy=True, episodes_eval=5)
            eval_queue.append(eval_reward)
            metrics["eval_mean_5"].append(np.mean(eval_queue))
        else:
            metrics["eval_mean_5"].append(metrics["eval_mean_5"][-1] if metrics["eval_mean_5"] else 0.0)

        if ep % 100 == 0 or ep == 1:
            print(f"Ep {ep}/{episodes} reward={total_reward:.2f} len={length} eps={epsilon:.4f} eval_mean5={metrics['eval_mean_5'][-1]:.2f}")

    # save q-table & metrics
    with open(qtable_path, "wb") as f:
        pickle.dump(dict(Q), f)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length", "epsilon", "eval_mean_5"])
        for i in range(len(metrics["episode"])):
            writer.writerow([metrics["episode"][i], metrics["reward"][i], metrics["length"][i], metrics["epsilon"][i], metrics["eval_mean_5"][i]])

    # plot learning curve (raw + smoothed)
    plt.figure(figsize=(9,4))
    episodes_list = metrics["episode"]
    plt.plot(episodes_list, metrics["reward"], alpha=0.3, label="Episode reward (raw)")
    window = 50 if len(episodes_list) >= 50 else max(1, len(episodes_list) // 2)
    if len(episodes_list) >= window:
        weights = np.ones(window) / window
        smoothed = np.convolve(metrics["reward"], weights, mode='valid')
        x_smoothed = episodes_list[window - 1:]
        plt.plot(x_smoothed, smoothed, color='tab:orange', label=f"Smoothed (window={window})")
    # also plot eval mean
    plt.plot(episodes_list, metrics["eval_mean_5"], label="Eval mean (last 5)", linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Compact Pacman learning curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print("Saved:", plot_path, qtable_path, csv_path)

def evaluate_policy(env, Q, obs_to_state_fn, greedy=True, episodes_eval=3):
    """
    Run a few episodes with greedy policy (or epsilon=0.1 if greedy=False).
    Returns mean total reward.
    """
    rewards = []
    for _ in range(episodes_eval):
        raw_obs, _ = env.reset()
        s = obs_to_state_fn(env, raw_obs)
        total = 0.0
        for _ in range(500):
            if greedy:
                a = int(np.argmax(Q[s]))
            else:
                if np.random.rand() < 0.1:
                    a = env.action_space.sample()
                else:
                    a = int(np.argmax(Q[s]))
            raw_obs, r, term, trunc, info = env.step(a)
            total += r
            s = obs_to_state_fn(env, raw_obs)
            if term or trunc:
                break
        rewards.append(total)
    return float(np.mean(rewards))

if __name__ == "__main__":
    # usage: python train_q_learning_compact.py [episodes] [grid]
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    grid = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    ghosts = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    out = f"logs_compact_{episodes}eps_{grid}grid_{ghosts}g"
    train(episodes=episodes, grid_size=grid, num_ghosts=ghosts, out_dir=out)
