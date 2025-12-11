# train_q_learning.py
import numpy as np
import pickle, csv, os, sys
from pacman_env import PacmanEnv
from collections import defaultdict
import matplotlib.pyplot as plt

# try importing config for GRID_SIZE; fall back if missing
try:
    from config import GRID_SIZE, NUM_GHOSTS
except Exception:
    GRID_SIZE = 21
    NUM_GHOSTS = 1

def moving_average(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")

def obs_to_state(obs):
    # obs is small numpy vector; convert to tuple of ints for Q-table keys
    return tuple(int(x) for x in obs)

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_q_learning.py <episodes>")
        return
    episodes = int(sys.argv[1])

    grid_size = GRID_SIZE
    num_ghosts = NUM_GHOSTS

    env = PacmanEnv(grid_size=grid_size, num_ghosts=num_ghosts)
    env.seed(0)

    Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float32))

    # hyperparams
    alpha = 0.03
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.9995

    # sensible max_steps
    free_cells = int(np.sum(env.walls == 0))
    factor = 3.0
    max_steps = int(max(300, min(2000, int(factor * free_cells))))
    print(f"Computed max_steps={max_steps} from free_cells={free_cells}")

    log_dir = f"logs/{episodes}_episodes"
    os.makedirs(log_dir, exist_ok=True)

    metrics = {"episode": [], "reward": [], "length": [], "epsilon": [], "pellets": [], "wins": []}

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state = obs_to_state(obs)
        total_reward = 0.0
        length = 0
        pellets_eaten = 0

        prev_pellet_count = int(np.sum(env.pellets))

        terminated = False
        truncated = False

        for t in range(max_steps):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            obs, reward, terminated, truncated, _ = env.step(action)

            # pellet counting via env (not via reward)
            curr_pellet_count = int(np.sum(env.pellets))
            eaten = prev_pellet_count - curr_pellet_count
            if eaten > 0:
                pellets_eaten += eaten
            prev_pellet_count = curr_pellet_count

            done = bool(terminated or truncated)
            next_state = obs_to_state(obs)

            # Q update
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

        # log
        metrics["episode"].append(ep)
        metrics["reward"].append(total_reward)
        metrics["length"].append(length)
        metrics["epsilon"].append(epsilon)
        metrics["pellets"].append(pellets_eaten)
        metrics["wins"].append(1 if int(np.sum(env.pellets)) == 0 else 0)

        if ep % 100 == 0 or ep == 1:
            recent = 100
            avg_reward = np.mean(metrics["reward"][-recent:]) if len(metrics["reward"]) >= recent else np.mean(metrics["reward"])
            avg_pellets = np.mean(metrics["pellets"][-recent:]) if len(metrics["pellets"]) >= recent else np.mean(metrics["pellets"])
            win_rate = np.sum(metrics["wins"][-recent:]) / recent if len(metrics["wins"]) >= recent else np.sum(metrics["wins"]) / max(1, len(metrics["wins"]))
            print(f"Ep {ep}/{episodes} reward={total_reward:.1f} pellets={pellets_eaten} len={length} eps={epsilon:.4f} avg_reward_100={avg_reward:.2f} avg_pellets_100={avg_pellets:.2f} win_rate_100={win_rate:.2f}")

    # save Q-table
    q_path = os.path.join(log_dir, f"q_table_{episodes}.pkl")
    with open(q_path, "wb") as f:
        pickle.dump(dict(Q), f)

    # save metrics CSV
    csv_path = os.path.join(log_dir, f"metrics_{episodes}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length", "epsilon", "pellets", "win"])
        for i in range(len(metrics["episode"])):
            writer.writerow([metrics["episode"][i], metrics["reward"][i], metrics["length"][i], metrics["epsilon"][i], metrics["pellets"][i], metrics["wins"][i]])

    # plotting
    window = 200
    episodes_arr = np.array(metrics["episode"])
    # 1) pellets
    pellets_arr = np.array(metrics["pellets"])
    pellets_sm = moving_average(pellets_arr, window)
    plt.figure(figsize=(8,4))
    plt.plot(episodes_arr, pellets_arr, alpha=0.25, label="raw pellets")
    if len(pellets_sm) > 0:
        plt.plot(episodes_arr[window-1:], pellets_sm, label=f"smoothed pellets (w={window})", linewidth=2)
    plt.xlabel("Episode"); plt.ylabel("Pellets eaten"); plt.grid(True); plt.legend()
    plt.title("Pellets eaten per episode")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"learning_pellets_{episodes}.png"))
    plt.close()

    # 2) rewards bar
    rewards_arr = np.array(metrics["reward"])
    rewards_sm = moving_average(rewards_arr, window)
    plt.figure(figsize=(10,4))
    plt.bar(episodes_arr, rewards_arr, alpha=0.4, label="reward per episode")
    if len(rewards_sm) > 0:
        plt.plot(episodes_arr[window-1:], rewards_sm, color="red", linewidth=2, label=f"smoothed reward (w={window})")
    plt.xlabel("Episode"); plt.ylabel("Total reward"); plt.title("Rewards per episode")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"rewards_bar_{episodes}.png"))
    plt.close()

    # 3) win rate
    wins_arr = np.array(metrics["wins"])
    win_rate_sm = moving_average(wins_arr, window)
    plt.figure(figsize=(8,4))
    if len(win_rate_sm) > 0:
        plt.plot(episodes_arr[window-1:], win_rate_sm, label=f"win rate (w={window})", linewidth=2)
    plt.xlabel("Episode"); plt.ylabel("Win rate"); plt.title("Win rate (moving average)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"winrate_{episodes}.png"))
    plt.close()

    # 4) epsilon decay
    eps_arr = np.array(metrics["epsilon"])
    plt.figure(figsize=(8,4))
    plt.plot(episodes_arr, eps_arr, linewidth=2)
    plt.xlabel("Episode"); plt.ylabel("Epsilon"); plt.title("Epsilon decay")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"epsilon_decay_{episodes}.png"))
    plt.close()

    # 5) heatmap of avg max-Q per pac position (average over other features)
    n = grid_size
    heatmap = np.full((n, n), np.nan, dtype=float)
    # iterate pac cells
    # possible ranges for other features:
    g_range = range(-env.clip_dist, env.clip_dist + 1)
    p_range = range(-env.clip_dist, env.clip_dist + 1)
    wall_opts = [0, 1]
    danger_opts = [0, 1]
    bucket_opts = [0,1,2,3]

    for pr in range(n):
        for pc in range(n):
            vals = []
            for gdx in g_range:
                for gdy in g_range:
                    for pdx in p_range:
                        for pdy in p_range:
                            for danger in danger_opts:
                                for wu in wall_opts:
                                    for wd in wall_opts:
                                        for wl in wall_opts:
                                            for wr in wall_opts:
                                                # build state tuple matching obs_to_state ordering
                                                st = (pr, pc, int(gdx), int(gdy), int(pdx), int(pdy), int(danger), int(wu), int(wd), int(wl), int(wr))
                                                if st in Q:
                                                    vals.append(np.max(Q[st]))
            heatmap[pr, pc] = float(np.mean(vals)) if vals else np.nan

    plt.figure(figsize=(6,6))
    plt.imshow(heatmap, origin="upper")
    plt.colorbar(label="avg max Q")
    plt.title("Heatmap: Avg max-Q per Pacman position")
    plt.xlabel("col"); plt.ylabel("row")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"q_heatmap_{episodes}.png"))
    plt.close()

    print("Saved results and Q-table in", log_dir)

if __name__ == "__main__":
    main()
