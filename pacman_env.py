# pacman_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# try to import config values if user has config.py, otherwise use defaults
try:
    from config import GRID_SIZE, NUM_GHOSTS, SKIP_WALLS
except Exception:
    GRID_SIZE = 21
    NUM_GHOSTS = 1
    SKIP_WALLS = False

class PacmanEnv(gym.Env):
    """
    Grid-based Pacman environment with compact observation.

    Observation (11 ints):
      [pac_r, pac_c,
       ghost_dx_clipped, ghost_dy_clipped,
       pellet_dx_clipped, pellet_dy_clipped,
       danger_flag,
       wall_up, wall_down, wall_left, wall_right]

    - ghost/pellet dx/dy are clipped to [-5..5].
    - danger_flag = 1 if nearest ghost manhattan distance <= 2 else 0
    - walls flags are 0/1 for immediate neighbor cells or 1 at grid border
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=GRID_SIZE, num_ghosts=NUM_GHOSTS, walls=None, clip_dist=5):
        super().__init__()
        self.grid_size = int(grid_size)
        self.num_ghosts = int(num_ghosts)
        self.clip_dist = int(clip_dist)
        self._fixed_walls_cache = None
        if walls is not None:
            walls = np.array(walls, dtype=np.int32)
            if walls.shape != (self.grid_size, self.grid_size):
                raise ValueError("Provided walls must match grid_size.")
            self._fixed_walls_cache = walls.copy()
        else:
            self._fixed_walls_cache = self._create_pacman_maze()
        self.action_space = spaces.Discrete(4)

        # observation bounds
        low = np.array([0, 0, -self.clip_dist, -self.clip_dist, -self.clip_dist, -self.clip_dist, 0, 0, 0, 0, 0], dtype=np.int32)
        high = np.array([self.grid_size - 1, self.grid_size - 1, self.clip_dist, self.clip_dist, self.clip_dist, self.clip_dist, 1, 1, 1, 1, 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self._init_positions()
        self.seed()
        self.reset()

    def _init_positions(self):
        # Pacman starts near center-bottom
        self.start_pac = [self.grid_size - 2, self.grid_size // 2]
        # Ghosts start in top gap areas; take as many as num_ghosts
        mid = self.grid_size // 2
        default_ghosts = [
            [1, mid - 1],
            [1, mid + 1],
            [1, mid],
            [2, mid - 2],
            [2, mid + 2],
        ]
        self.start_ghosts = default_ghosts[: self.num_ghosts]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pacman = self.start_pac.copy()
        self.ghosts = np.array(self.start_ghosts, dtype=np.int32)
        self.walls = self._fixed_walls_cache.copy()
        # ensure start cells are open
        self.walls[self.start_pac[0], self.start_pac[1]] = 0
        for gr, gc in self.start_ghosts:
            self.walls[gr, gc] = 0
        self.pellets = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        self.pellets[self.pacman[0], self.pacman[1]] = 0
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

        max_steps = 2000
        reward = -0.05
        hit_wall_reward = -1.0
        eat_pellet_reward = 20.0
        ghost_catch_reward = -200.0
        all_pellets_eaten_reward = 500.0
        step_limit_reward = -0.1

        terminated = False
        truncated = False

        # wall collision check for target (if wall, penalize and don't move)
        if self.walls[target[0], target[1]] == 1:
            reward += hit_wall_reward
        else:
            self.pacman = target

        # eat pellet
        if self.pellets[self.pacman[0], self.pacman[1]] == 1:
            reward += eat_pellet_reward
            self.pellets[self.pacman[0], self.pacman[1]] = 0

        # ghosts move randomly (blocked by walls)
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

        # collision check
        if any((self.pacman[0] == int(g[0]) and self.pacman[1] == int(g[1])) for g in self.ghosts):
            reward += ghost_catch_reward
            terminated = True

        # all pellets eaten
        if np.sum(self.pellets) == 0:
            reward += all_pellets_eaten_reward
            terminated = True

        # step limit
        if self.steps >= max_steps:
            reward += step_limit_reward
            truncated = True

        obs = self._get_obs()
        info = {}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _get_obs(self):
        pac_r, pac_c = int(self.pacman[0]), int(self.pacman[1])

        # nearest ghost (Manhattan)
        min_dist = 10**9
        nearest = (pac_r, pac_c)
        for gr, gc in self.ghosts:
            d = int(abs(int(gr) - pac_r) + abs(int(gc) - pac_c))
            if d < min_dist:
                min_dist = d
                nearest = (int(gr), int(gc))
        dx_g = nearest[0] - pac_r
        dy_g = nearest[1] - pac_c
        dx_g = int(np.clip(dx_g, -self.clip_dist, self.clip_dist))
        dy_g = int(np.clip(dy_g, -self.clip_dist, self.clip_dist))

        # nearest pellet (Manhattan)
        pellet_positions = np.argwhere(self.pellets == 1)
        if pellet_positions.shape[0] == 0:
            px, py = 0, 0
        else:
            dists = np.abs(pellet_positions - np.array([pac_r, pac_c])).sum(axis=1)
            idx = np.argmin(dists)
            pr, pc = pellet_positions[idx]
            px = int(np.clip(int(pr) - pac_r, -self.clip_dist, self.clip_dist))
            py = int(np.clip(int(pc) - pac_c, -self.clip_dist, self.clip_dist))

        # danger flag
        danger = 1 if min_dist <= 2 else 0

        # wall indicators (treat border as wall)
        if pac_r - 1 < 0:
            wall_up = 1
        else:
            wall_up = 1 if self.walls[pac_r - 1, pac_c] == 1 else 0
        if pac_r + 1 >= self.grid_size:
            wall_down = 1
        else:
            wall_down = 1 if self.walls[pac_r + 1, pac_c] == 1 else 0
        if pac_c - 1 < 0:
            wall_left = 1
        else:
            wall_left = 1 if self.walls[pac_r, pac_c - 1] == 1 else 0
        if pac_c + 1 >= self.grid_size:
            wall_right = 1
        else:
            wall_right = 1 if self.walls[pac_r, pac_c + 1] == 1 else 0

        obs = np.array([
            pac_r, pac_c,
            dx_g, dy_g,
            px, py,
            danger,
            wall_up, wall_down, wall_left, wall_right
        ], dtype=np.int32)

        return obs

    def _create_pacman_maze(self):
        walls = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Outer border
        walls[0, :] = 1
        walls[-1, :] = 1
        walls[:, 0] = 1
        walls[:, -1] = 1

        if not SKIP_WALLS:
            mid = self.grid_size // 2  # 10 for 21x21

            # -------- TRUE PLUS SIGN IN THE CENTER --------
            # Horizontal arms (leave a wide center gap so all quadrants connect)
            walls[mid, 3:9] = 1     # left arm (cols 3..8)
            walls[mid, 13:19] = 1   # right arm (cols 13..18)
            # Vertical arms
            walls[3:9, mid] = 1     # top arm (rows 3..8)
            walls[13:19, mid] = 1   # bottom arm (rows 13..18)
            # The center block rows 9..12 and cols 9..12 remain open (wide center opening)

            # -------- LARGE CORRIDORS / RINGS (with big center gaps) --------
            # Horizontal corridor walls (rows 5 and 15) but leave wide center opening
            walls[5, 1:-1] = 1
            walls[5, 7:14] = 0     # open span across the middle (cols 7..13)
            walls[15, 1:-1] = 1
            walls[15, 7:14] = 0

            # Vertical corridor walls (cols 5 and 15) but leave wide center opening
            walls[1:-1, 5] = 1
            walls[7:14, 5] = 0     # open span across the middle (rows 7..13)
            walls[1:-1, 15] = 1
            walls[7:14, 15] = 0

            # -------- CORNER CONNECTIONS (prevent closed chambers) --------
            # Make small openings that connect corner regions to the main corridors
            # (these ensure there are no sealed rooms in corners)
            walls[1, 5] = 0
            walls[2, 5] = 0
            walls[5, 2] = 0
            walls[5, 1] = 0
            walls[1, 15] = 0
            walls[2, 15] = 0
            walls[15, 1] = 0
            walls[15, 2] = 0
            walls[5, 18] = 0
            walls[5, 19] = 0
            walls[18, 5] = 0
            walls[19, 5] = 0
            walls[18, 15] = 0
            walls[18, 10] = 0
            walls[19, 15] = 0
            walls[15, 18] = 0
            walls[15, 19] = 0
            walls[5, 6] = 0
            walls[6, 5] = 0
            walls[5, 14] = 0
            walls[6, 15] = 0
            walls[5, 14] = 0
            walls[14, 5] = 0
            walls[15, 14] = 0
            walls[15, 6] = 0
            walls[14, 15] = 0
            walls[18, 11] = 0
            walls[10, 12] = 1
            walls[10, 18] = 0
            walls[12, 10] = 1
        
        return walls

    def render(self, mode="rgb_array"):
        cell = 20
        W = self.grid_size * cell
        img = 255 * np.ones((W, W, 3), dtype=np.uint8)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.walls[r, c] == 1:
                    r0, c0 = r * cell, c * cell
                    img[r0:r0 + cell, c0:c0 + cell] = (20, 20, 20)
                elif self.pellets[r, c] == 1:
                    cy = int((r + 0.5) * cell)
                    cx = int((c + 0.5) * cell)
                    rr = slice(cy - 2, cy + 3)
                    cc = slice(cx - 2, cx + 3)
                    img[rr, cc] = (80, 80, 80)

        pr, pc = self.pacman
        r0, c0 = pr * cell, pc * cell
        img[r0 + 2:r0 + cell - 2, c0 + 2:c0 + cell - 2] = (255, 220, 0)

        for gr, gc in self.ghosts:
            r0, c0 = int(gr) * cell, int(gc) * cell
            img[r0 + 2:r0 + cell - 2, c0 + 2:c0 + cell - 2] = (200, 0, 0)

        for k in range(1, self.grid_size):
            img[k * cell - 1:k * cell + 1, :, :] = 220
            img[:, k * cell - 1:k * cell + 1, :] = 220

        return img

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]
