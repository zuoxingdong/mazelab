import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors

from .Astar_solver import AstarSolver

import gym
from gym import spaces
from gym.utils import seeding

class GridWorldEnv(gym.Env):
    """Configurable environment for grid world. """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_generator, pob_size=1, trace=False, action_type='VonNeumann'):
        """Initialize the grid world with a given map. DType: list"""
        # Grid map: 0: free space, 1: wall
        self.grid_generator = grid_generator
        self.grid_map = np.array(self.grid_generator.get())
        self.grid_size = self.grid_map.shape
        self.init_state, self.goal_states = self.grid_generator.sample_state()
        
        self.trace = trace
        self.action_type = action_type
        
        self.state = None
        
        # Action space: 0: Up, 1: Down, 2: Left, 3: Right
        if self.action_type == 'VonNeumann':  # Von Neumann neighborhood
            self.num_actions = 4
        elif action_type == 'Moore':  # Moore neighborhood
            self.num_actions = 8
        else:
            raise TypeError('Action type must be either \'VonNeumann\' or \'Moore\'')
        self.action_space = spaces.Discrete(self.num_actions)
        self.all_actions = list(range(self.action_space.n))
        # Observation space: Tuple of rows
        low_obs = 0  # Lowest integer in observation
        high_obs = 6  # Highest integer in observation
        self.observation_space = spaces.Tuple(
                        [spaces.MultiDiscrete(np.tile([low_obs, high_obs], [self.grid_size[1], 1])), ]*self.grid_size[0]
                        )
        
        # Size of the partial observable window
        self.pob_size = pob_size
        
        # Create Figure for rendering
        self.fig, (self.ax_full, self.ax_partial) = plt.subplots(nrows=1, ncols=2)
        # Colormap: order of color is, free space, wall, agent, food, poison
        self.cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray'])
        self.bounds = [0, 1, 2, 3, 4, 5, 6]  # values for each color
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        self.ax_full_img = self.ax_full.imshow(self.grid_map, cmap=self.cmap, norm=self.norm, animated=True)
        
        self.ax_full.axis('off')
        self.ax_partial.axis('off')
        
        self.ax_imgs = []  # For generating videos
        
    def _step(self, action):
        old_state = self.state
        # Update current state
        self.state = self._next_state(self.state, action)
        
        # Footprint: Change state to indicate agent's trajectory
        if self.trace:
            self.grid_map[self.state[0], self.state[1]] = 6
        
        if self._goal_test(self.state):  # Goal check
            reward = +1
            done = True
        elif self.state == old_state:  # Hit wall
            reward = -1
            done = False
        else:  # Moved, small negative reward to encourage shorest path
            reward = -0.01
            done = False
        
        # Additional info
        info = {}
        
        return self._get_obs(), reward, done, info
    
    def _reset(self, solve=False):
        # Reset grid map
        self.grid_map = np.array(self.grid_generator.get())
        
        # Set current state be initial state
        self.state = self.init_state
        
        # Compute optimal trajectories by A* search from initial position to each goal
        self.optimal_solution = {}
        if solve:
            for i, goal in enumerate(self.goal_states):
                solver = AstarSolver(self, goal)

                if solver.solvable():
                    self.optimal_solution[('optimal actions', i)] = solver.get_actions()
                    self.optimal_solution[('optimal trajectory', i)] = solver.get_states()
        
        # Clean the list of ax_imgs, the buffer for generating videos
        self.ax_imgs = []
        
        return self._get_obs()
    
    def _render(self, mode='human', close=False):
        obs = self._get_obs()
        partial_obs = self._get_partial_obs(self.pob_size)
        
        self.fig.show()
        self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
        self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)
        
        plt.draw()
        
        # Put in AxesImage buffer for video generation
        self.ax_imgs.append([self.ax_full_img, self.ax_partial_img])  # List of axes to update figure frame
        
        self.fig.set_dpi(100)
        return self.fig
        
    def _goal_test(self, state):
        """Return True if current state is a goal state."""
        if type(self.goal_states[0]) == list:
            return list(state) in self.goal_states
        elif type(self.goal_states[0]) == tuple:
            return tuple(state) in self.goal_states
    
    def _next_state(self, state, action):
        """Return the next state from a given state by taking a given action."""
        
        # Transition table to define movement for each action
        if self.action_type == 'VonNeumann':
            transitions = {0: [-1, 0], 1: [+1, 0], 2: [0, -1], 3: [0, +1]}
        elif self.action_type == 'Moore':
            transitions = {0: [-1, 0], 1: [+1, 0], 2: [0, -1], 3: [0, +1], 
                           4: [-1, +1], 5: [+1, +1], 6: [-1, -1], 7: [+1, -1]}
        
        new_state = [state[0] + transitions[action][0], state[1] + transitions[action][1]]
        if self.grid_map[new_state[0]][new_state[1]] == 1:  # Hit wall, stay there
            return state
        else:  # Valid move for 0, 2, 3, 4
            return new_state
            
    def _step_cost(self, state, action, next_state):
        """Return a cost that a given action leads a state to a next_state"""
        return 1  # Simple grid world: uniform cost for each step in the path.
    
    def _get_obs(self):
        """Return a 2D array representation of grid world."""
        obs = np.array(self.grid_map)
        # Set goal positions
        for goal in self.goal_states:
            obs[goal[0]][goal[1]] = 3  # 3: goal
        # Set current position
        # Come after painting goal positions, avoid invisible within multi-goal regions
        obs[self.state[0]][self.state[1]] = 2  # 2: agent
        
        return obs
    
    def _get_partial_obs(self, size=1):
        """Get partial observable window according to Moore neighborhood"""
        # Get grid map with indicated location of current position and goal positions
        grid = self._get_obs()
        pos = np.array(self.state)

        under_offset = np.min(pos - size)
        over_offset = np.min(len(grid) - (pos + size + 1))
        offset = np.min([under_offset, over_offset])

        if offset < 0:  # Need padding
            grid = np.pad(grid, np.abs(offset), 'constant', constant_values=1)
            pos += np.abs(offset)

        return grid[pos[0]-size : pos[0]+size+1, pos[1]-size : pos[1]+size+1]
        
    def _get_video(self, interval=200, gif_path=None):
        anim = animation.ArtistAnimation(self.fig, self.ax_imgs, interval=interval)
        
        if gif_path is not None:
            anim.save(gif_path, writer='imagemagick', fps=10)
        return anim