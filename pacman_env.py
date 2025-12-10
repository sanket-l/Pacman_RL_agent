# pacman_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PacmanEnv(gym.Env):
    """
    Simple grid-based Pacman-like environment for RL experiments.
    Grid contains:
      - pacman (agent)
      - three ghosts (adversaries)
      - randomly placed walls (impassable)
      - pellets (1 or 0) on each non-wall cell (except pacman start)
    Actions: 0=UP,1=DOWN,2=LEFT,3=RIGHT
    Observation: [pac_x, pac_y, ghost_coords..., pellets_flat, walls_flat]
      ghost_coords is the concatenation of all ghost (x, y) pairs.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=10, num_ghosts=3, wall_density=0.15, fix_walls=True, walls=None):
        super().__init__()
        self.grid_size = grid_size
        self.num_ghosts = num_ghosts
        self.wall_density = wall_density
        self.fix_walls = fix_walls
        self.initial_walls = None
        self._fixed_walls_cache = None
        if walls is not None:
            walls = np.array(walls, dtype=np.int32)
            if walls.shape != (grid_size, grid_size):
                raise ValueError("Provided walls must match grid_size.")
            self.initial_walls = walls
            self._fixed_walls_cache = walls.copy()
            self.fix_walls = True  # ensure consistent reuse when walls supplied
        self.action_space = spaces.Discrete(4)
        # obs: pac (2) + ghosts (2*num_ghosts) + pellets + walls (each grid_size*grid_size)
        self.observation_space = spaces.Box(
            low=0,
            high=grid_size - 1,
            shape=(
                2 + 2 * num_ghosts + grid_size * grid_size + grid_size * grid_size,
            ),
            dtype=np.int32,
        )
        self._init_positions()
        self.reset()

    def _init_positions(self):
        self.start_pac = [0, 0]
        # spread ghosts across remaining corners (avoids pacman start cell)
        self.start_ghosts = [
            [self.grid_size - 1, self.grid_size - 1],
            [self.grid_size - 1, 0],
            [0, self.grid_size - 1],
        ][: self.num_ghosts]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pacman = self.start_pac.copy()
        self.ghosts = np.array(self.start_ghosts, dtype=np.int32)
        if self.fix_walls and self._fixed_walls_cache is not None:
            self.walls = self._fixed_walls_cache.copy()
        else:
            self.walls = self._generate_walls()
            if self.fix_walls:
                self._fixed_walls_cache = self.walls.copy()
        # ensure start cells are open even if provided walls had them blocked
        self.walls[self.start_pac[0], self.start_pac[1]] = 0
        for gr, gc in self.start_ghosts:
            self.walls[gr, gc] = 0
        self.pellets = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        self.pellets[self.pacman[0], self.pacman[1]] = 0
        # pellets cannot exist on walls
        self.pellets[self.walls == 1] = 0
        self.steps = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.steps += 1
        # apply pacman action
        target = list(self.pacman)
        if action == 0:  # up
            target[0] = max(0, self.pacman[0] - 1)
        elif action == 1:  # down
            target[0] = min(self.grid_size - 1, self.pacman[0] + 1)
        elif action == 2:  # left
            target[1] = max(0, self.pacman[1] - 1)
        elif action == 3:  # right
            target[1] = min(self.grid_size - 1, self.pacman[1] + 1)
        # block on walls
        if self.walls[target[0], target[1]] == 0:
            self.pacman = target

        reward = 0
        terminated = False
        truncated = False

        # eat pellet if present
        if self.pellets[self.pacman[0], self.pacman[1]] == 1:
            reward += 10
            self.pellets[self.pacman[0], self.pacman[1]] = 0

        # ghosts move randomly (simple baseline), blocked by walls
        for g in range(len(self.ghosts)):
            move = self.np_random.integers(0, 4)
            target_g = list(self.ghosts[g])
            if move == 0:
                target_g[0] = max(0, self.ghosts[g, 0] - 1)
            elif move == 1:
                target_g[0] = min(self.grid_size - 1, self.ghosts[g, 0] + 1)
            elif move == 2:
                target_g[1] = max(0, self.ghosts[g, 1] - 1)
            elif move == 3:
                target_g[1] = min(self.grid_size - 1, self.ghosts[g, 1] + 1)
            if self.walls[target_g[0], target_g[1]] == 0:
                self.ghosts[g] = target_g

        # collision check: any ghost catches pacman
        if any((self.pacman[0] == g[0] and self.pacman[1] == g[1]) for g in self.ghosts):
            reward -= 200
            terminated = True

        # all pellets eaten
        if np.sum(self.pellets) == 0:
            reward += 100
            terminated = True

        # optional step limit to avoid infinite episodes
        if self.steps >= 1000:
            truncated = True

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        pac = np.array(self.pacman, dtype=np.int32)
        ghosts = self.ghosts.flatten().astype(np.int32)
        pellets = self.pellets.flatten().astype(np.int32)
        walls = self.walls.flatten().astype(np.int32)
        return np.concatenate([pac, ghosts, pellets, walls])

    def _generate_walls(self):
        """
        Randomly generate walls with the given density, ensuring start cells are free.
        """
        walls = (self.np_random.random((self.grid_size, self.grid_size)) < self.wall_density).astype(np.int32)
        # ensure start positions are free
        walls[self.start_pac[0], self.start_pac[1]] = 0
        for gr, gc in self.start_ghosts:
            walls[gr, gc] = 0
        return walls

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
                if self.walls[r, c] == 1:
                    r0, c0 = r * cell, c * cell
                    img[r0:r0 + cell, c0:c0 + cell] = (20, 20, 20)  # dark wall
                elif self.pellets[r, c] == 1:
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

        # ghost squares (red)
        for gr, gc in self.ghosts:
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
