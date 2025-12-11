# pacman_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PacmanEnv(gym.Env):
    """
    Compact Pacman-like environment designed for tabular Q-learning.
    - Default grid_size small (11)
    - Pellets are present but NOT encoded in full observation
    - Observation returned to agent is a full raw vector (for flexibility),
      but we provide a compact obs_to_state() helper in the training script.
    - Ghosts move greedily toward Pacman (deterministic baseline).
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=11, num_ghosts=2, fixed_walls=None):
        super().__init__()
        assert grid_size >= 7 and grid_size % 2 == 1, "use odd grid_size >= 7"
        self.grid_size = grid_size
        self.num_ghosts = num_ghosts
        self._fixed_walls_cache = None
        if fixed_walls is not None:
            walls = np.array(fixed_walls, dtype=np.int32)
            if walls.shape != (grid_size, grid_size):
                raise ValueError("Provided walls must match grid_size.")
            self._fixed_walls_cache = walls.copy()
        else:
            self._fixed_walls_cache = self._create_simple_maze()

        self.action_space = spaces.Discrete(4)  # up, down, left, right

        # Raw observation (for debugging) - pac + ghosts + pellets + walls flattened
        raw_len = 2 + 2 * self.num_ghosts + grid_size * grid_size + grid_size * grid_size
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(raw_len,), dtype=np.int32)

        self._init_positions()
        self.reset()

    def _init_positions(self):
        self.start_pac = [self.grid_size - 2, self.grid_size // 2]
        mid = self.grid_size // 2
        # place ghosts near the top center area
        starts = [[1, mid - 1], [1, mid + 1], [2, mid]]
        self.start_ghosts = starts[: self.num_ghosts]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pacman = self.start_pac.copy()
        self.ghosts = np.array(self.start_ghosts, dtype=np.int32)
        self.walls = self._fixed_walls_cache.copy()
        # ensure start cells are open
        self.walls[self.pacman[0], self.pacman[1]] = 0
        for gr, gc in self.start_ghosts:
            self.walls[gr, gc] = 0

        # pellets: 1 in every open cell except pacman start
        self.pellets = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        self.pellets[self.walls == 1] = 0
        self.pellets[self.pacman[0], self.pacman[1]] = 0

        self.steps = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.steps += 1
        # pacman move attempt
        target = list(self.pacman)
        if action == 0:  # up
            target[0] = max(0, self.pacman[0] - 1)
        elif action == 1:  # down
            target[0] = min(self.grid_size - 1, self.pacman[0] + 1)
        elif action == 2:  # left
            target[1] = max(0, self.pacman[1] - 1)
        elif action == 3:  # right
            target[1] = min(self.grid_size - 1, self.pacman[1] + 1)

        reward = 0.0
        terminated = False
        truncated = False

        # small penalty per move to encourage efficiency
        reward += -0.1

        # hitting wall -> stay in place and penalty
        if self.walls[target[0], target[1]] == 1:
            reward += -0.5
        else:
            self.pacman = target

        # eat pellet
        if self.pellets[self.pacman[0], self.pacman[1]] == 1:
            reward += 5.0
            self.pellets[self.pacman[0], self.pacman[1]] = 0

        # ghosts move: greedy move that reduces Manhattan distance (deterministic)
        for g_idx in range(len(self.ghosts)):
            gx, gy = int(self.ghosts[g_idx, 0]), int(self.ghosts[g_idx, 1])
            pr, pc = self.pacman

            # prefer move along the larger coordinate difference
            dr = pr - gx
            dc = pc - gy
            moves = []
            if abs(dr) >= abs(dc):
                # vertical preference
                if dr < 0:
                    moves.append((-1, 0))
                elif dr > 0:
                    moves.append((1, 0))
                # then try horizontal
                if dc < 0:
                    moves.append((0, -1))
                elif dc > 0:
                    moves.append((0, 1))
            else:
                # horizontal preference
                if dc < 0:
                    moves.append((0, -1))
                elif dc > 0:
                    moves.append((0, 1))
                if dr < 0:
                    moves.append((-1, 0))
                elif dr > 0:
                    moves.append((1, 0))

            # add fallback moves to allow movement if a greedy direction is blocked
            moves += [(-1,0),(1,0),(0,-1),(0,1)]

            moved = False
            for d_r, d_c in moves:
                tg_r, tg_c = gx + d_r, gy + d_c
                if tg_r < 0 or tg_r >= self.grid_size or tg_c < 0 or tg_c >= self.grid_size:
                    continue
                if self.walls[tg_r, tg_c] == 0:
                    self.ghosts[g_idx] = [tg_r, tg_c]
                    moved = True
                    break
            if not moved:
                # stuck - stay in place
                pass

        # collision check
        if any((self.pacman[0] == int(g[0]) and self.pacman[1] == int(g[1])) for g in self.ghosts):
            reward += -100.0
            terminated = True

        # win condition - all pellets eaten
        if np.sum(self.pellets) == 0:
            reward += 200.0
            terminated = True

        # simple timeout
        if self.steps >= 500:
            truncated = True

        obs = self._get_obs()
        info = {}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _get_obs(self):
        pac = np.array(self.pacman, dtype=np.int32)
        ghosts = self.ghosts.flatten().astype(np.int32)
        pellets = self.pellets.flatten().astype(np.int32)
        walls = self.walls.flatten().astype(np.int32)
        return np.concatenate([pac, ghosts, pellets, walls])

    def _create_simple_maze(self):
        # smaller symmetric simple maze (no enclosed chambers)
        s = self.grid_size
        walls = np.zeros((s, s), dtype=np.int32)
        walls[0,:] = 1
        walls[-1,:] = 1
        walls[:,0] = 1
        walls[:,-1] = 1
        mid = s // 2

        # cross-shaped walls with gaps
        walls[mid, 2:mid-1] = 1
        walls[mid, mid+2:-2] = 1
        walls[2:mid-1, mid] = 1
        walls[mid+2:-2, mid] = 1

        # carve openings so corners connect
        walls[2, mid] = 0
        walls[mid, 2] = 0
        walls[-3, mid] = 0
        walls[mid, -3] = 0

        return walls

    def render(self, mode="rgb_array"):
        cell = 20
        W = self.grid_size * cell
        img = 255 * np.ones((W, W, 3), dtype=np.uint8)
        # draw walls
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.walls[r,c] == 1:
                    img[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = (30,30,30)
                elif self.pellets[r,c] == 1:
                    cy = int((r + 0.5) * cell)
                    cx = int((c + 0.5) * cell)
                    rr = slice(cy-2, cy+3)
                    cc = slice(cx-2, cx+3)
                    img[rr, cc] = (80,80,80)
        # pacman
        pr, pc = self.pacman
        img[pr*cell+2:(pr+1)*cell-2, pc*cell+2:(pc+1)*cell-2] = (255,220,0)
        # ghosts
        for gr, gc in self.ghosts:
            img[int(gr)*cell+2:(int(gr)+1)*cell-2, int(gc)*cell+2:(int(gc)+1)*cell-2] = (200,0,0)
        # grid lines
        for k in range(1, self.grid_size):
            img[k*cell-1:k*cell+1,:,:] = 220
            img[:,k*cell-1:k*cell+1,:] = 220
        return img

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]
