# train_q_learning_compact.py
import numpy as np
import pickle, csv, os, sys
from pacman_env import PacmanEnv
from collections import defaultdict, deque
import matplotlib.pyplot as plt

# ----------------------
# compact state encoder
# ----------------------
def sign3(x):
    """Return -1,0,1 for negative, zero, positive."""
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0

def obs_to_compact_state(obs, grid_size, num_ghosts):
    """
    Map the full obs vector to a small discrete state tuple:
    (pac_r, pac_c,
      for each ghost: dx_clipped + 3, dy_clipped + 3,
      pellet_dir_r + 1, pellet_dir_c + 1,   # mapped to {0,1,2}
      pellets_bucket)
    dx_clipped in [-3..3] mapped to 0..6
    pellet_dir in {-1,0,1} mapped to 0..2
    pellets_bucket in {0,1,2,3}
    """
    # obs layout: [pac(2), ghosts(2*num_ghosts), pellets(grid^2), walls(grid^2)]
    pac_r = int(obs[0])
    pac_c = int(obs[1])
    ghosts_offset = 2
    pellets_offset = 2 + 2 * num_ghosts
    pellets_flat = obs[pellets_offset: pellets_offset + grid_size * grid_size].astype(np.int32)
    pellets_grid = pellets_flat.reshape((grid_size, grid_size))

    # ghosts relative positions clipped
    ghost_components = []
    for g in range(num_ghosts):
        gx = int(obs[ghosts_offset + 2 * g + 0])
        gy = int(obs[ghosts_offset + 2 * g + 1])
        dx = gx - pac_r
        dy = gy - pac_c
        # clip to [-3, 3]
        dx = max(-3, min(3, dx))
        dy = max(-3, min(3, dy))
        ghost_components.append(dx + 3)  # 0..6
        ghost_components.append(dy + 3)  # 0..6

    # nearest pellet direction (sign of row diff, col diff)
    pellet_positions = np.argwhere(pellets_grid == 1)
    if pellet_positions.shape[0] == 0:
        pdr, pdc = 0, 0
    else:
        # choose nearest by manhattan distance
        dists = np.abs(pellet_positions - np.array([pac_r, pac_c])).sum(axis=1)
        idx = np.argmin(dists)
        pr, pc = pellet_positions[idx]
        pdr = sign3(pr - pac_r)   # -1,0,1
        pdc = sign3(pc - pac_c)   # -1,0,1

    # remaining pellets bucket (coarse)
    remaining = pellets_positions_count = int(pellets_grid.sum())
    if remaining == 0:
        bucket = 0
    elif remaining <= 5:
        bucket = 1
    elif remaining <= 20:
        bucket = 2
    else:
        bucket = 3

    # build tuple (all ints, small range)
    state = (pac_r, pac_c) + tuple(ghost_components) + (pdr + 1, pdc + 1, bucket)
    return tuple(int(x) for x in state)

# ----------------------
# training script
# ----------------------
def moving_average(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_q_learning.py <episodes>")
        return

    episodes = int(sys.argv[1])

    # create env (keep defaults or change)
    grid_size = 21
    num_ghosts = 1   # set to match your env usage
    env = PacmanEnv(grid_size=grid_size, num_ghosts=num_ghosts)
    env.seed(0)

    # Q-table
    Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float32))

    # hyperparams (conservative stable choices)
    alpha = 0.03
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.9995   # slow multiplicative decay

    # sensible max_steps derived from free cells
    free_cells = int(np.sum(env._fixed_walls_cache == 0))
    factor = 3.0
    max_steps = int(max(300, min(2000, int(factor * free_cells))))
    print(f"Computed max_steps={max_steps} from free_cells={free_cells}")

    log_dir = f"logs/{episodes}_episodes"
    os.makedirs(log_dir, exist_ok=True)

    metrics = {"episode": [], "reward": [], "length": [], "epsilon": [], "pellets": [], "wins": []}

    # training loop
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state = obs_to_compact_state(obs, grid_size, num_ghosts)
        total_reward = 0.0
        length = 0
        pellets_eaten = 0

        # pellet counting via grid diff
        pellets_offset = 2 + 2 * num_ghosts
        prev_pellet_grid = obs[pellets_offset:pellets_offset + grid_size * grid_size].copy().astype(np.int32).reshape((grid_size, grid_size))
        initial_pellets = int(prev_pellet_grid.sum())

        terminated = False
        truncated = False

        for t in range(max_steps):
            # epsilon-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            obs, reward, terminated, truncated, _ = env.step(action)

            # count pellets eaten by grid diff
            curr_pellet_grid = obs[pellets_offset:pellets_offset + grid_size * grid_size].astype(np.int32).reshape((grid_size, grid_size))
            eaten = int(prev_pellet_grid.sum() - curr_pellet_grid.sum())
            if eaten > 0:
                pellets_eaten += eaten
            prev_pellet_grid = curr_pellet_grid

            done = bool(terminated or truncated)
            next_state = obs_to_compact_state(obs, grid_size, num_ghosts)

            # Q update with terminal handling
            best_next = float(np.max(Q[next_state])) if next_state in Q else 0.0
            if done:
                td_target = float(reward)
            else:
                td_target = float(reward) + gamma * best_next
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state
            total_reward += float(reward)
            length += 1
            if done:
                break

        # decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # record metrics
        metrics["episode"].append(ep)
        metrics["reward"].append(total_reward)
        metrics["length"].append(length)
        metrics["epsilon"].append(epsilon)
        metrics["pellets"].append(pellets_eaten)
        metrics["wins"].append(1 if np.sum(prev_pellet_grid) == 0 else 0)

        # periodic prints
        if ep % 100 == 0 or ep == 1:
            recent = 100
            avg_reward = np.mean(metrics["reward"][-recent:]) if len(metrics["reward"]) >= recent else np.mean(metrics["reward"])
            avg_pellets = np.mean(metrics["pellets"][-recent:]) if len(metrics["pellets"]) >= recent else np.mean(metrics["pellets"])
            win_rate = np.sum(metrics["wins"][-recent:]) / recent if len(metrics["wins"]) >= recent else np.sum(metrics["wins"]) / max(1, len(metrics["wins"]))
            print(f"Ep {ep}/{episodes} reward={total_reward:.1f} pellets={pellets_eaten} len={length} eps={epsilon:.4f} avg_reward_100={avg_reward:.2f} avg_pellets_100={avg_pellets:.2f} win_rate_100={win_rate:.2f}")

    # save Q-table
    with open(os.path.join(log_dir, f"q_table_{episodes}.pkl"), "wb") as f:
        pickle.dump(dict(Q), f)

    # save metrics CSV
    csv_path = os.path.join(log_dir, f"metrics_{episodes}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length", "epsilon", "pellets", "win"])
        for i in range(len(metrics["episode"])):
            writer.writerow([metrics["episode"][i], metrics["reward"][i], metrics["length"][i], metrics["epsilon"][i], metrics["pellets"][i], metrics["wins"][i]])

    # plotting: smoothed pellets and win-rate
    window = 200
    episodes_arr = np.array(metrics["episode"])
    pellets_arr = np.array(metrics["pellets"])
    pellets_sm = moving_average(pellets_arr, window)

    wins_arr = np.array(metrics["wins"])
    win_rate_sm = moving_average(wins_arr, window)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(episodes_arr, pellets_arr, alpha=0.25, label="raw pellets")
    if len(pellets_sm) > 0:
        plt.plot(episodes_arr[window-1:], pellets_sm, label=f"smoothed pellets (w={window})", linewidth=2)
    plt.xlabel("Episode"); plt.ylabel("Pellets eaten"); plt.grid(True); plt.legend()

    plt.subplot(1,2,2)
    if len(win_rate_sm) > 0:
        plt.plot(episodes_arr[window-1:], win_rate_sm, label=f"win rate (w={window})")
    plt.xlabel("Episode"); plt.ylabel("Win rate"); plt.grid(True); plt.legend()

    plt.suptitle("Learning (compact-state Q-learning)")
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(os.path.join(log_dir, f"learning_{episodes}.png"))
    print("Saved results in", log_dir)

if __name__ == "__main__":
    main()
