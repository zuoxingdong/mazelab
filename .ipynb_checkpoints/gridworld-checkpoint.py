import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors

from Astar_solver import AstarSolver
from utils import RandomGridGenerator


class GridWorldEnv(object):
    """Configurable environment for grid world. """
    
    def __init__(self, grid_map, init_state, goal_states, pob_size=1):
        """Initialize the grid world with a given map. DType: list"""
        # Grid map: 0: free space, 1: wall
        self.grid_map = np.array(grid_map)
        self.init_state = init_state
        self.goal_states = goal_states  # A list of multiple (or single) goal states
        
        self.state = None
        self.all_actions = [0, 1, 2, 3]  # 0: Up, 1: Down, 2: Left, 3: Right
        
        self.pob_size = pob_size  # size of the partial observable window
        
        # Create Figure for rendering
        self.fig, (self.ax_full, self.ax_partial) = plt.subplots(nrows=1, ncols=2)
        # Colormap: order of color is, free space, wall, agent, food, poison
        self.cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red'])
        self.bounds = [0, 1, 2, 3, 4, 5]  # values for each color
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        self.ax_full_img = self.ax_full.imshow(self.grid_map, cmap=self.cmap, norm=self.norm, animated=True)
        
        self.ax_full.axis('off')
        self.ax_partial.axis('off')
        
        self.ax_imgs = []  # For generating videos
        
    def _step(self, action):
        old_state = self.state
        # Update current state
        self.state = self._next_state(self.state, action)
        
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
    
    def _reset(self, solve=True):
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
        return state in self.goal_states
    
    def _next_state(self, state, action):
        """Return the next state from a given state by taking a given action."""
        
        # Transition table to define movement for each action
        transitions = {0: [-1, 0], 1: [+1, 0], 2: [0, -1], 3: [0, +1]}
        
        new_state = [state[0] + transitions[action][0], state[1] + transitions[action][1]]
        if self.grid_map[new_state[0]][new_state[1]] == 0:  # Valid move
            return new_state
        else:  # Hit wall, stay there
            return state
            
    def _step_cost(self, state, action, next_state):
        """Return a cost that a given action leads a state to a next_state"""
        return 1  # Simple grid world: uniform cost for each step in the path.
    
    def _get_obs(self):
        """Return a 2D array representation of grid world."""
        obs = np.array(self.grid_map)
        # Set current position
        obs[self.state[0]][self.state[1]] = 2  # 2: agent
        # Set goal positions
        for goal in self.goal_states:
            obs[goal[0]][goal[1]] = 3  # 3: goal
        
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