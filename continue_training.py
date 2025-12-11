#!/usr/bin/env python3
"""Continue training from a saved Q-table.

Usage examples:
  python3 continue_training.py --qpath logs/100_episodes/q_table_100.pkl --add 200
  python3 continue_training.py --qpath logs/100_episodes/q_table_100.pkl --add 500 --alpha 0.05

The script will load the pickled Q (a dict mapping state tuples to numpy arrays),
wrap it into a defaultdict with zeros for unseen states (matching the current
environment action space), then continue Q-learning for `--add` episodes and
save updated Q and metrics to the same directory (or to `--outdir`).
"""
import argparse
import glob
import os
import pickle
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

from pacman_env import PacmanEnv


def obs_to_state(obs):
    return tuple(int(x) for x in obs)


def load_qtable(path, action_n):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # data is expected to be a dict mapping state tuples -> arrays
    Q = defaultdict(lambda: np.zeros(action_n))
    for k, v in data.items():
        arr = np.array(v)
        # if loaded Q has different action dim, warn and replace with zeros
        if arr.shape[0] != action_n:
            print(f"Warning: Q entry for state {k} has length {arr.shape[0]} != env action_n {action_n}; replacing with zeros")
            Q[k] = np.zeros(action_n)
        else:
            Q[k] = arr
    return Q


def save_qtable(Q, out_path):
    # convert defaultdict to plain dict for portability
    data = {k: v for k, v in Q.items()}
    with open(out_path, "wb") as f:
        pickle.dump(data, f)


def run(args):
    # reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create env -- use same default as your training script (adjust args.env_grid if needed)
    env = PacmanEnv(grid_size=args.grid_size)
    env.seed(args.seed)

    action_n = env.action_space.n

    if not os.path.exists(args.qpath):
        raise FileNotFoundError(f"Q-table not found: {args.qpath}")

    Q = load_qtable(args.qpath, action_n)

    # If requested, try to infer last epsilon from previous metrics in same dir
    if args.resume_metrics:
        metrics_dir = os.path.dirname(args.qpath) or '.'
        # look for metrics_*.csv or metrics_continued_*.csv
        candidates = glob.glob(os.path.join(metrics_dir, 'metrics_*.csv')) + glob.glob(os.path.join(metrics_dir, 'metrics_continued_*.csv'))
        if candidates:
            # pick most recently modified
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            metrics_file = candidates[0]
            try:
                with open(metrics_file, newline='') as mf:
                    rdr = list(csv.reader(mf))
                    if len(rdr) >= 2:
                        last_row = rdr[-1]
                        # header may include epsilon at last column
                        header = [h.strip().lower() for h in rdr[0]]
                        if 'epsilon' in header:
                            eps_idx = header.index('epsilon')
                            last_eps = float(last_row[eps_idx])
                            epsilon = last_eps
                            print(f"Resuming epsilon from {metrics_file}: epsilon={epsilon}")
                        else:
                            print(f"No 'epsilon' column in {metrics_file}; keeping provided epsilon={epsilon}")
                    else:
                        print(f"Metrics file {metrics_file} empty; keeping provided epsilon={epsilon}")
            except Exception as e:
                print(f"Failed to read metrics file {metrics_file}: {e}; keeping provided epsilon={epsilon}")
        else:
            print(f"No metrics file found in {metrics_dir}; keeping provided epsilon={epsilon}")

    # hyperparams (can be overridden)
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon
    epsilon_min = args.epsilon_min
    epsilon_decay = args.epsilon_decay
    max_steps = args.max_steps

    outdir = args.outdir or os.path.dirname(args.qpath)
    os.makedirs(outdir, exist_ok=True)

    metrics = {"episode": [], "reward": [], "length": [], "epsilon": []}

    for ep in range(1, args.add + 1):
        obs, _ = env.reset()
        state = obs_to_state(obs)
        total_reward = 0
        length = 0

        for t in range(max_steps):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = obs_to_state(obs)

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

        if ep % max(1, args.add // 10) == 0 or ep == 1:
            print(f"Cont Ep {ep}/{args.add} reward={total_reward:.1f} len={length} eps={epsilon:.3f}")

    # save updated Q-table and metrics
    base_name = os.path.splitext(os.path.basename(args.qpath))[0]
    q_out_name = f"{base_name}_continued_{args.add}.pkl"
    q_out_path = os.path.join(outdir, q_out_name)
    save_qtable(Q, q_out_path)

    csv_path = os.path.join(outdir, f"metrics_continued_{args.add}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length", "epsilon"])
        for i in range(len(metrics["episode"])):
            writer.writerow([metrics["episode"][i], metrics["reward"][i], metrics["length"][i], metrics["epsilon"][i]])

    # plot
    plt.figure(figsize=(8, 4))
    episodes = metrics["episode"]
    plt.plot(episodes, metrics["reward"], alpha=0.5, label="Episode reward (raw)")
    window = 50 if len(episodes) >= 50 else max(1, len(episodes) // 2)
    if len(episodes) >= window:
        weights = np.ones(window) / window
        smoothed = np.convolve(metrics["reward"], weights, mode='valid')
        x_smoothed = episodes[window - 1:]
        plt.plot(x_smoothed, smoothed, color='tab:orange', label=f"Smoothed (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Continued learning curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(outdir, f"learning_curve_continued_{args.add}.png")
    plt.savefig(plot_path)

    print(f"Saved continued Q-table to {q_out_path}")
    print(f"Saved metrics to {csv_path} and plot to {plot_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--qpath", required=True, help="Path to existing q_table .pkl file")
    p.add_argument("--add", type=int, required=True, help="Number of additional episodes to run")
    p.add_argument("--outdir", help="Directory to save continued Q-table and metrics (default: same dir as qpath)")
    p.add_argument("--resume-metrics", action="store_true", help="If set, attempt to read previous metrics CSV in qpath's dir and resume epsilon from its last value")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon_min", type=float, default=0.05)
    p.add_argument("--epsilon_decay", type=float, default=0.9995)
    p.add_argument("--max_steps", type=int, default=750)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid_size", type=int, default=21, help="Grid size for PacmanEnv (match original training)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
