# pacman_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PacmanEnv(gym.Env):
    """
    Simple grid-based Pacman-like environment for RL experiments.
    Grid contains:
      - pacman (agent)
      - one ghost (adversary)
      - pellets (1 or 0) on each cell (except pacman start)
    Actions: 0=UP,1=DOWN,2=LEFT,3=RIGHT
    Observation: concatenation of [pac_x, pac_y, ghost_x, ghost_y, pellets_flat]
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=7):
        super().__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)
        # obs: 4 position ints + grid_size*grid_size pellets (0/1)
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, shape=(4 + grid_size * grid_size,), dtype=np.int32
        )
        self._init_positions()
        self.reset()

    def _init_positions(self):
        self.start_pac = [0, 0]
        self.start_ghost = [self.grid_size - 1, self.grid_size - 1]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pacman = self.start_pac.copy()
        self.ghost = self.start_ghost.copy()
        self.pellets = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        self.pellets[self.pacman[0], self.pacman[1]] = 0
        self.steps = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.steps += 1
        # apply pacman action
        if action == 0:  # up
            self.pacman[0] = max(0, self.pacman[0] - 1)
        elif action == 1:  # down
            self.pacman[0] = min(self.grid_size - 1, self.pacman[0] + 1)
        elif action == 2:  # left
            self.pacman[1] = max(0, self.pacman[1] - 1)
        elif action == 3:  # right
            self.pacman[1] = min(self.grid_size - 1, self.pacman[1] + 1)

        reward = 0
        terminated = False
        truncated = False

        # eat pellet if present
        if self.pellets[self.pacman[0], self.pacman[1]] == 1:
            reward += 10
            self.pellets[self.pacman[0], self.pacman[1]] = 0

        # ghost moves randomly (simple baseline)
        move = self.np_random.integers(0, 4)
        if move == 0:
            self.ghost[0] = max(0, self.ghost[0] - 1)
        elif move == 1:
            self.ghost[0] = min(self.grid_size - 1, self.ghost[0] + 1)
        elif move == 2:
            self.ghost[1] = max(0, self.ghost[1] - 1)
        elif move == 3:
            self.ghost[1] = min(self.grid_size - 1, self.ghost[1] + 1)

        # collision check
        if self.pacman == self.ghost:
            reward -= 200
            terminated = True

        # all pellets eaten
        if np.sum(self.pellets) == 0:
            reward += 100
            terminated = True

        # optional step limit to avoid infinite episodes
        if self.steps >= 300:
            truncated = True

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        pac = np.array(self.pacman, dtype=np.int32)
        ghost = np.array(self.ghost, dtype=np.int32)
        pellets = self.pellets.flatten().astype(np.int32)
        return np.concatenate([pac, ghost, pellets])

    def render(self, mode="rgb_array"):
        """
        Returns an RGB numpy array representing the grid.
        Colors:
          - background: white
          - pellet: small dark dot
          - pacman: yellow square
          - ghost: red square
        """
        cell = 20  # pixels per cell
        W = self.grid_size * cell
        img = 255 * np.ones((W, W, 3), dtype=np.uint8)

        # draw pellets
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.pellets[r, c] == 1:
                    cy = int((r + 0.5) * cell)
                    cx = int((c + 0.5) * cell)
                    # small dot
                    rr = slice(cy - 2, cy + 3)
                    cc = slice(cx - 2, cx + 3)
                    img[rr, cc] = (80, 80, 80)  # dark dot

        # pacman square (yellow)
        pr, pc = self.pacman
        r0, c0 = pr * cell, pc * cell
        img[r0 + 2:r0 + cell - 2, c0 + 2:c0 + cell - 2] = (255, 220, 0)

        # ghost square (red)
        gr, gc = self.ghost
        r0, c0 = gr * cell, gc * cell
        img[r0 + 2:r0 + cell - 2, c0 + 2:c0 + cell - 2] = (200, 0, 0)

        # grid lines
        for k in range(1, self.grid_size):
            img[k * cell - 1:k * cell + 1, :, :] = 220
            img[:, k * cell - 1:k * cell + 1, :] = 220

        return img

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]
