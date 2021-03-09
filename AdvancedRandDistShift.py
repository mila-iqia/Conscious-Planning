import numpy as np
from gym_minigrid.minigrid import *

class AdvancedRandDistShift(MiniGridEnv):

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(self, width=8, height=8, lava_density_range=[0.3, 0.4], min_num_route=1, transposed=False, gamma=0.99):
        lava_density = np.random.uniform(lava_density_range[0], lava_density_range[1])
        self.min_num_route = min_num_route
        self.transposed = transposed
        if self.transposed:
            self.total_possible_lava = (width * height - 2 * height)
        else:
            self.total_possible_lava = (width * height - 2 * width)
        self.max_lava_blocks = int(self.total_possible_lava * lava_density)
        self.agent_start_dir = np.random.randint(0, 4)
        if self.transposed:
            if np.random.rand() <= 0.5:
                self.agent_start_pos = (np.random.randint(1, width), 0)
                self.goal_pos = (np.random.randint(0, width - 1), height - 1)
            else:
                self.agent_start_pos = (np.random.randint(0, width), height - 1)
                self.goal_pos = (np.random.randint(0, width), 0)
        else:
            if np.random.rand() <= 0.5:
                self.agent_start_pos = (0, np.random.randint(0, height))
                self.goal_pos = (width - 1, np.random.randint(0, height))
            else:
                self.agent_start_pos = (width - 1, np.random.randint(0, height))
                self.goal_pos = (0, np.random.randint(0, height))

        self.rand_width = width
        self.rand_height = height
        self.generate_map()
        super().__init__(width=width, height=height, max_steps=4 * width * height, see_through_walls=True, agent_view_size=15) # Set this to True for maximum speed
        self.actions = AdvancedRandDistShift.Actions
        self.num_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.num_actions)
        self.Q_optimal = None
        self.gamma = gamma

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.width, self.height, 3),  # number of cells
            dtype='uint8'
        )
        self.obs_curr = self.gen_fullyobservable_obs()
        
    def attention_render(self, highlight=True, tile_size=TILE_PIXELS, highlight_mask=None, C_first=True):
        """
        Render the whole-grid human view
        """
        # Mask of which cells to highlight
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)
        else:
            highlight_mask = highlight_mask.reshape(self.height, self.width)

        # Render the whole grid
        img = self.att_render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None
        )

        if C_first: img = np.transpose(img, (2, 0, 1))
        return img

    def att_render(
        self,
        tile_size,
        agent_pos=None,
        agent_dir=None,
        highlight_mask=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.grid.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = self.att_render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def att_render_tile(
        self,
        obj,
        agent_dir=None,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in self.grid.tile_cache:
            return self.grid.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img_var(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        self.grid.tile_cache[key] = img

        return img


    def gen_fullyobservable_obs(self):
        full_grid = self.grid.encode()
        full_grid[self.agent_pos[0]][self.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            self.agent_dir
        ])
        return full_grid

    def generate_oracle(self):    # depends on action encoding
        # Reshape observation
        maps = self.get_DP_map(self.obs_curr)
        
        # Generate P (depends on action encoding)
        num_states = self.width * self.height * 4
        P = np.zeros([3, num_states, num_states])
        for i in range(self.height):
            for j in range(self.width):
                for k in range(4):
                    idx_state = self.ijd2state(i, j, k)

                    if maps[i,j] == 8 or maps[i,j] == 9:
                        P[:, idx_state, idx_state] = 1.0
                        continue

                    P[0, idx_state, self.ijd2state(i, j, ((k - 1) % 4))] = 1
                    P[1, idx_state, self.ijd2state(i, j, ((k + 1) % 4))] = 1

                    if k == 0:
                        if j!=self.width-1:
                            P[2, idx_state, self.ijd2state(i, j + 1, k)]=1
                        else:
                            P[2, idx_state, idx_state]=1
                    if k == 1:
                        if i!=self.height-1:
                            P[2, idx_state, self.ijd2state(i + 1, j, k)]=1
                        else:
                            P[2,idx_state,idx_state]=1
                    if k == 2:
                        if j != 0:
                            P[2,idx_state,self.ijd2state(i, j - 1, k)]=1
                        else:
                            P[2,idx_state,idx_state]=1
                    if k == 3:
                        if i != 0:
                            P[2,idx_state,self.ijd2state(i - 1, j, k)]=1
                        else:
                            P[2,idx_state,idx_state]=1

        #Generate r
        r = np.zeros([num_states, self.num_actions])
        goal_found = False
        for i in range(self.height):
            if goal_found: break
            for j in range(self.width):
                if maps[i, j] == 8:
                    goal_i, goal_j = i, j
                    goal_found = True
                    break
        if goal_j != self.width - 1 and maps[goal_i, goal_j+1] != 9:
            r[self.ijd2state(goal_i, goal_j + 1, 2), 2] = 1
        if goal_i != self.height - 1 and maps[goal_i + 1, goal_j] != 9:
            r[self.ijd2state(goal_i + 1, goal_j, 3), 2] = 1
        if goal_j != 0 and maps[goal_i, goal_j - 1] != 9:
            r[self.ijd2state(goal_i, goal_j - 1, 0), 2] = 1
        if goal_i != 0 and maps[goal_i - 1, goal_j] != 9:
            r[self.ijd2state(goal_i - 1, goal_j, 1), 2] = 1

        VmulP = lambda v, P: np.matmul(P, v).transpose()
        v0 = np.zeros(num_states)
        Boper = lambda r, P, v: np.max(r + self.gamma * VmulP(v, P), axis=-1)
        v_old = v0
        while True:
            v_new = Boper(r, P, v_old)
            if np.sum(np.abs(v_new - v_old)) <= 1e-7: break
            v_old = v_new
        self.Q_optimal = r + self.gamma * VmulP(v_new, P)

    def evaluate_action(self, action, obs=None):
        if obs is None: obs = self.obs_curr
        if self.Q_optimal is None: self.generate_oracle()
        return float(action in self.get_optimal_actions(self.get_DP_state(obs)))

    def evaluate_action_extra(self, obs=None):
        if obs is None: obs = self.obs_curr
        if self.Q_optimal is None: self.generate_oracle()
        list_quality_actions = []
        q = self.Q_optimal[self.get_DP_state(obs), :].squeeze()
        q_max = np.max(q)
        list_optimal_actions = np.where(q == q_max)[0].tolist()
        for action in range(self.action_space.n):
            list_quality_actions.append(float(action in list_optimal_actions))
        return list_quality_actions, q

    def get_optimal_actions(self, state):
        q = self.Q_optimal[state, :].squeeze()
        q_max = np.max(q)
        return np.where(q == q_max)[0].tolist()

    @staticmethod
    def get_DP_map(obs):
        maps = obs[:, :, 0] + obs[:, :, 2]
        return maps.squeeze().transpose()

    def get_DP_state(self, obs):
        maps = self.get_DP_map(obs)
        agent_found = False
        for i in range(self.height):
            if agent_found: break
            for j in range(self.width):
                d = maps[i, j] - 10
                if d >= 0:
                    agent_found = True
                    agent_i, agent_j = i, j
                    break
        if not agent_found: raise ValueError
        return self.ijd2state(agent_i, agent_j, d)

    def ijd2state(self, i, j, d):
        return i * 4 * self.width + j * 4 + d

    def generate_random_path(self, epsilon=0.35):
        goal = self.goal_pos
        current_state = np.array(self.agent_start_pos)
        duration = 0
        while True:
            if duration == 0:
                duration = np.random.randint(1, 4)
                difference_x, difference_y = goal[0] - current_state[0], goal[1] - current_state[1]
                x_rand, y_rand = False, False
                action_list, random_action_list = [], []
                if difference_x != 0:
                    direction_diff_x = int(np.sign(difference_x))
                    action_list.append([direction_diff_x, 0]); random_action_list.append([-direction_diff_x, 0])
                else:
                    random_action_list.append([np.random.randint(0, 1) * 2 - 1, 0])
                    x_rand = True
                
                if difference_y != 0:
                    direction_diff_y = int(np.sign(difference_y))
                    action_list.append([0, direction_diff_y]); random_action_list.append([0, -direction_diff_y])
                else:
                    random_action_list.append([0, np.random.randint(0, 1) * 2 - 1])
                    y_rand = True

            if np.random.uniform(0, 1) > epsilon:
                if len(action_list) == 0:
                    break
                else:
                    current_action = action_list[int(np.random.randint(0, len(action_list)))]
            else:
                if x_rand:
                    current_action = random_action_list[0]
                elif y_rand:
                    current_action = random_action_list[1]
                else:
                    current_action = random_action_list[int(np.random.randint(0, len(random_action_list)))]
            current_state[0] += current_action[0]
            current_state[1] += current_action[1]
            current_state[0] = np.clip(current_state[0], 0, self.rand_width - 1)
            current_state[1] = np.clip(current_state[1], 0, self.rand_height - 1)
            self.test_grid[current_state[0], current_state[1]] = 0
            duration -= 1
            if current_state[0] == goal[0] and current_state[1] == goal[1]: break

    def reset_gen_map(self):
        self.test_grid = np.zeros((self.rand_width, self.rand_height))
        if self.transposed:
            self.test_grid[0: self.rand_width, 1: self.rand_height - 1] = 1
        else:
            self.test_grid[1: self.rand_width - 1, 0: self.rand_height] = 1
        self.test_grid[self.agent_start_pos[0], self.agent_start_pos[1]] = 0
        self.test_grid[self.goal_pos[0], self.goal_pos[1]] = 0
    
    def generate_map(self):
        self.reset_gen_map()
        while True:
            for i in range(self.min_num_route):
                self.generate_random_path()
            remaining_lava_blocks = int(np.sum(self.test_grid))
            if remaining_lava_blocks > self.max_lava_blocks:
                break
            self.reset_gen_map()

        if remaining_lava_blocks > self.max_lava_blocks:
            lava_indices = np.nonzero(self.test_grid)
            lava_indices_x = lava_indices[0]
            lava_indices_y = lava_indices[1]
            perm = np.random.permutation(lava_indices_x.shape[0])
            lava_indices_x = lava_indices_x[perm]
            lava_indices_y = lava_indices_y[perm]
            for i in range(int(remaining_lava_blocks - self.max_lava_blocks)):
                self.test_grid[lava_indices_x[i], lava_indices_y[i]] = 0

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        # self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), *self.goal_pos)

        for i in range(0, self.test_grid.shape[0]):
            for j in range(0, self.test_grid.shape[1]):
                if self.test_grid[i, j] == 1:
                    self.grid.set(i, j, Lava())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def reset(self):
        super().reset()
        self.obs_curr = self.gen_fullyobservable_obs()
        return self.obs_curr

    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        flag_inside = True
        if self.front_pos[0] < 0 or self.front_pos[0] >= self.width: flag_inside = False
        if self.front_pos[1] < 0 or self.front_pos[1] >= self.height: flag_inside = False
        fwd_cell = self.grid.get(*fwd_pos) if flag_inside else None
        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        # Move forward
        elif action == self.actions.forward:
            if flag_inside:
                if fwd_cell == None or fwd_cell.can_overlap():
                    self.agent_pos = fwd_pos
                if fwd_cell != None and fwd_cell.type == 'goal':
                    done = True
                    reward = self._reward()
                if fwd_cell != None and fwd_cell.type == 'lava':
                    done = True
        else:
            assert False, "unknown action"
        if self.step_count >= self.max_steps:
            done = True
        self.obs_curr = self.gen_fullyobservable_obs()
        return self.obs_curr, np.sign(reward), done, {}

def highlight_img_var(img, color=(255, 255, 255), alpha=0.30):
    """
    Add highlighting to an image
    """
    blend_img = (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img