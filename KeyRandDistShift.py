"""
DEFINITION FILE OF KEYDOOR (UNLOCK) ENVIRONMENTS (W/ TURN-AND-FORWARD DYNAMICS)
"""
import numpy as np
from gym_minigrid.minigrid import *
from RandDistShift import highlight_img_var

class KeyRandDistShift(MiniGridEnv):

    class Actions(IntEnum):
        left_forward = 0
        forward = 1
        right_forward = 2
        back_forward = 3

    def __init__(self, width=8, height=8, lava_density_range=[0.3, 0.4], min_num_route=1, transposed=False, gamma=0.99, random_color=False):
        lava_density = np.random.uniform(lava_density_range[0], lava_density_range[1])
        self.key_acquired = False
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
        self.random_color = bool(random_color)
        super().__init__(width=width, height=height, max_steps=4 * width * height, see_through_walls=True, agent_view_size=15) # Set this to True for maximum speed
        self.actions = KeyRandDistShift.Actions
        self.num_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.num_actions)
        self.Q_optimal = None
        self.solvable = False
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
        if self.random_color:
            mask_randcolor = np.random.randint(6, size=(self.width, self.height))
            full_grid[:, :, 1] = mask_randcolor # set random color range(5) to all grids
        full_grid[self.agent_pos[0]][self.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            self.agent_dir
        ])
        return full_grid

    def clear_path2key(self, epsilon=0.35):
        goal = self.key_pos
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
            self.test_grid[current_state[0], current_state[1]] = 0 # erase stuffs
            duration -= 1
            if current_state[0] == goal[0] and current_state[1] == goal[1]: break

    def clear_path2goal(self, epsilon=0.35):
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
            self.test_grid[current_state[0], current_state[1]] = 0 # erase stuffs
            duration -= 1
            if current_state[0] == goal[0] and current_state[1] == goal[1]: break

    def reset_gen_map(self):
        self.test_grid = np.zeros((self.rand_width, self.rand_height))
        if self.transposed:
            self.test_grid[0: self.rand_width, 1: self.rand_height - 1] = 1
            self.key_pos = (np.random.randint(0, self.rand_width), np.random.randint(1, self.rand_height - 1))
        else:
            self.test_grid[1: self.rand_width - 1, 0: self.rand_height] = 1
            self.key_pos = (np.random.randint(1, self.rand_width - 1), np.random.randint(0, self.rand_height))
        self.test_grid[self.agent_start_pos[0], self.agent_start_pos[1]] = 0
        self.test_grid[self.goal_pos[0], self.goal_pos[1]] = 0
        self.test_grid[self.key_pos[0], self.key_pos[1]] = 0
    
    def generate_map(self):
        self.reset_gen_map()
        while True:
            for i in range(self.min_num_route):
                self.clear_path2key()
                self.clear_path2goal()
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
        self.put_obj(Key(), *self.key_pos)

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
        self.key_acquired = False
        self.obs_curr = self.gen_fullyobservable_obs()
        return self.obs_curr

    def check_inside(self, pos):
        flag_inside = True
        if pos[0] < 0 or pos[0] >= self.width: flag_inside = False
        if pos[1] < 0 or pos[1] >= self.height: flag_inside = False
        return flag_inside

    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False
        if action == self.actions.left_forward:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
        elif action == self.actions.right_forward:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == self.actions.back_forward:
            self.agent_dir = (self.agent_dir + 2) % 4
        elif action == self.actions.forward:
            pass
        else:
            assert False, "unknown action"
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        flag_inside = self.check_inside(fwd_pos)
        if flag_inside:
            fwd_cell = self.grid.get(*fwd_pos) if flag_inside else None
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                if self.key_acquired:
                    done = True
                    reward = 0.5
                else:
                    done = False
                    reward = 0
            if fwd_cell != None and fwd_cell.type == 'key':
                if self.key_acquired:
                    done = False
                    reward = 0
                else:
                    self.agent_pos = fwd_pos
                    self.put_obj(Floor(), *self.key_pos) # a colored floor
                    done = False
                    reward = 0.5
                    self.key_acquired = True
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True
        if self.step_count >= self.max_steps:
            done = True
        self.obs_curr = self.gen_fullyobservable_obs()
        return self.obs_curr, reward, done, {}