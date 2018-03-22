import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors

import gym
from gym import spaces
from gym.utils import seeding


class MazeEnv(gym.Env):
    """Configurable environment for maze. """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 maze_generator, 
                 pob_size=1,
                 action_type='VonNeumann',
                 obs_type='full',
                 live_display=False,
                 render_trace=False):
        """Initialize the maze. DType: list"""
        # Random seed with internal gym seeding
        self.seed()
        
        # Maze: 0: free space, 1: wall
        self.maze_generator = maze_generator
        self.maze = np.array(self.maze_generator.get_maze())
        self.maze_size = self.maze.shape
        self.init_state, self.goal_states = self.maze_generator.sample_state()
        
        self.render_trace = render_trace
        self.traces = []
        self.action_type = action_type
        self.obs_type = obs_type
        
        # If True, show the updated display each time render is called rather
        # than storing the frames and creating an animation at the end
        self.live_display = live_display

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

        # Size of the partial observable window
        self.pob_size = pob_size

        # Observation space
        low_obs = 0  # Lowest integer in observation
        high_obs = 6  # Highest integer in observation
        if self.obs_type == 'full':
            self.observation_space = spaces.Box(low=low_obs, 
                                                high=high_obs,
                                                shape=self.maze_size, 
                                                dtype=np.float32)
        elif self.obs_type == 'partial':
            self.observation_space = spaces.Box(low=low_obs, 
                                                high=high_obs,
                                                shape=(self.pob_size*2+1, self.pob_size*2+1), 
                                                dtype=np.float32)
        else:
            raise TypeError('Observation type must be either \'full\' or \'partial\'')
        
        
        # Colormap: order of color is, free space, wall, agent, food, poison
        self.cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray'])
        self.bounds = [0, 1, 2, 3, 4, 5, 6]  # values for each color
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        
        self.ax_imgs = []  # For generating videos
        
    def step(self, action):
        old_state = self.state
        # Update current state
        self.state = self._next_state(self.state, action)
        
        # Footprint: Record agent trajectory
        self.traces.append(self.state)
        
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
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        
        return [seed]
    
    def reset(self):
        # Reset maze
        self.maze = np.array(self.maze_generator.get_maze())
        
        # Set current state be initial state
        self.state = self.init_state
        
        # Clean the list of ax_imgs, the buffer for generating videos
        self.ax_imgs = []
        # Clean the traces of the trajectory
        self.traces = [self.init_state]
        
        return self._get_obs()
    
    def render(self, mode='human', close=False):
        if close:
            plt.close()
            return
        
        obs = self._get_full_obs()
        partial_obs = self._get_partial_obs(self.pob_size)
        
        # For rendering traces: Only for visualization, does not affect the observation data
        if self.render_trace:
            obs[list(zip(*self.traces[:-1]))] = 6
        
        # Create Figure for rendering
        if not hasattr(self, 'fig'):  # initialize figure and plotting axes
            self.fig, (self.ax_full, self.ax_partial) = plt.subplots(nrows=1, ncols=2)
        self.ax_full.axis('off')
        self.ax_partial.axis('off')
        
        self.fig.show()
        if self.live_display:
            # Only create the image the first time
            if not hasattr(self, 'ax_full_img'):
                self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            if not hasattr(self, 'ax_partial_img'):
                self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)
            # Update the image data for efficient live video
            self.ax_full_img.set_data(obs)
            self.ax_partial_img.set_data(partial_obs)
        else:
            # Create a new image each time to allow an animation to be created
            self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)
        
        plt.draw()
        
        if self.live_display:
            # Update the figure display immediately
            self.fig.canvas.draw()
        else:
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
        if self.maze[new_state[0]][new_state[1]] == 1:  # Hit wall, stay there
            return state
        else:  # Valid move for 0, 2, 3, 4
            return new_state
    
    def _get_obs(self):
        if self.obs_type == 'full':
            return self._get_full_obs()
        elif self.obs_type == 'partial':
            return self._get_partial_obs(self.pob_size)

    def _get_full_obs(self):
        """Return a 2D array representation of maze."""
        obs = np.array(self.maze)
        # Set goal positions
        for goal in self.goal_states:
            obs[goal[0]][goal[1]] = 3  # 3: goal
        
        # Set current position
        # Come after painting goal positions, avoid invisible within multi-goal regions
        obs[self.state[0]][self.state[1]] = 2  # 2: agent
        
        return obs
    
    def _get_partial_obs(self, size=1):
        """Get partial observable window according to Moore neighborhood"""
        # Get maze with indicated location of current position and goal positions
        maze = self._get_full_obs()
        pos = np.array(self.state)

        under_offset = np.min(pos - size)
        over_offset = np.min(len(maze) - (pos + size + 1))
        offset = np.min([under_offset, over_offset])

        if offset < 0:  # Need padding
            maze = np.pad(maze, np.abs(offset), 'constant', constant_values=1)
            pos += np.abs(offset)

        return maze[pos[0]-size : pos[0]+size+1, pos[1]-size : pos[1]+size+1]
        
    def _get_video(self, interval=200, gif_path=None):
        if self.live_display:
            # TODO: Find a way to create animations without slowing down the live display
            print('Warning: Generating an Animation when live_display=True not yet supported.')
        anim = animation.ArtistAnimation(self.fig, self.ax_imgs, interval=interval)
        
        if gif_path is not None:
            anim.save(gif_path, writer='imagemagick', fps=10)
        return anim
    

class SparseMazeEnv(MazeEnv):
    def step(self, action):
        obs, reward, done, info = super()._step(action)
        
        # Indicator reward function
        if reward != 1:
            reward = 0
            
        return obs, reward, done, info
            
        
##########################################
# TODO: Make Partial observable envs as OOP-style
###########################################
        


