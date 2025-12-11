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
      - symmetric Pacman-style maze walls (impassable, fixed layout)
      - pellets (1 or 0) on each non-wall cell (except pacman start)
    Actions: 0=UP,1=DOWN,2=LEFT,3=RIGHT
    Observation: [pac_x, pac_y, ghost_coords..., pellets_flat, walls_flat]
      ghost_coords is the concatenation of all ghost (x, y) pairs.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=21, num_ghosts=2, walls=None):
        super().__init__()
        self.grid_size = grid_size
        self.num_ghosts = num_ghosts
        self._fixed_walls_cache = None
        if walls is not None:
            walls = np.array(walls, dtype=np.int32)
            if walls.shape != (grid_size, grid_size):
                raise ValueError("Provided walls must match grid_size.")
            self._fixed_walls_cache = walls.copy()
        else:
            # Generate symmetric Pacman-style maze
            self._fixed_walls_cache = self._create_pacman_maze()
        self.action_space = spaces.Discrete(4)
        # obs: pac (2) + ghosts (2*num_ghosts) + pellets + walls (each grid_size*grid_size)
        self.observation_space = spaces.Box(
            low=0,
            high=np.array(
                [grid_size - 1] * (2 + 2 * num_ghosts) +
                [1] * (grid_size * grid_size) + 
                [1] * (grid_size * grid_size)
            ),
            dtype=np.int32,
        )
        self._init_positions()
        self.reset()

    def _init_positions(self):
        # Pacman starts near center-bottom (classic Pacman position)
        self.start_pac = [self.grid_size - 2, self.grid_size // 2]
        # Ghosts start in open corridor areas (top row, in gaps of horizontal walls)
        # These positions are in the gap areas where row 2 has openings, ensuring ghosts can move down
        mid = self.grid_size // 2
        self.start_ghosts = [
            [1, mid - 1],  # top-left of center gap (can move down through gap)
            [1, mid + 1],  # top-right of center gap (can move down through gap)
            [1, mid],      # top-center (in gap of horizontal wall)
        ][: self.num_ghosts]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pacman = self.start_pac.copy()
        self.ghosts = np.array(self.start_ghosts, dtype=np.int32)
        # Always use the fixed maze layout
        self.walls = self._fixed_walls_cache.copy()
        # ensure start cells are open
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
        
        max_steps = 2000
        reward = -0.05                 # per-step base (keep this)
        hit_wall_reward = -1           # punish trying to move into a wall
        eat_pellet_reward = 20         # keep pellet reward
        ghost_catch_reward = -200      # keep ghost penalty
        all_pellets_eaten_reward = 500 # keep win reward
        step_limit_reward = -0.1       # much smaller truncation penalty


        terminated = False
        truncated = False
        
        # negative reward for hitting walls
        if self.walls[target[0], target[1]] == 1:
            reward += hit_wall_reward  # penalty for trying to move into wall
        else:
            self.pacman = target

        # eat pellet if present
        if self.pellets[self.pacman[0], self.pacman[1]] == 1:
            reward += eat_pellet_reward
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
            reward += ghost_catch_reward
            terminated = True

        # all pellets eaten
        if np.sum(self.pellets) == 0:
            reward += all_pellets_eaten_reward
            terminated = True

        # optional step limit to avoid infinite episodes
        if self.steps >= max_steps:
            reward += step_limit_reward
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

    def _create_pacman_maze(self):
        """
        Create a simple symmetric maze with a plus shape in center, horizontal and vertical walls with gaps.
        Guarantees no closed chambers - every area has multiple exits.
        Returns a fixed maze layout that's the same every time.
        """
        # 21x21 symmetric Pacman-style layout (no closed chambers, real plus sign)
        walls = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Outer border
        walls[0, :] = 1
        walls[-1, :] = 1
        walls[:, 0] = 1
        walls[:, -1] = 1

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
