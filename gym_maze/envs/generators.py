import numpy as np
from numpy.random import randint

from itertools import product as cartesian_product

from skimage.draw import circle, circle_perimeter


class MazeGenerator(object):
    def __init__(self):
        self.maze = None
    
    def sample_state(self):
        """Uniformaly sample an initial state and a goal state"""
        # Get indices for all free spaces, i.e. zero
        free_space = np.where(self.maze == 0)
        free_space = list(zip(free_space[0], free_space[1]))

        # Sample indices for initial state and goal state
        init_idx, goal_idx = np.random.choice(len(free_space), size=2, replace=False)
        
        # Convert initial state to a list, goal states to list of list
        init_state = list(free_space[init_idx])
        goal_states = [list(free_space[goal_idx])]  # TODO: multiple goals
        
        return init_state, goal_states
        
    def get_maze(self):
        return self.maze


class SimpleMazeGenerator(MazeGenerator):
    def __init__(self, maze):
        super().__init__()
        
        self.maze = maze
        

class RandomMazeGenerator(MazeGenerator):
    def __init__(self, width=81, height=51, complexity=.75, density=.75):
        super().__init__()
        
        self.width = width
        self.height = height
        self.complexity = complexity
        self.density = density
        
        self.maze = self._generate_maze()
        
    def _generate_maze(self):
        """
        Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """
        # Only odd shapes
        shape = ((self.height // 2) * 2 + 1, (self.width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        self.complexity = int(self.complexity * (5 * (shape[0] + shape[1])))
        self.density    = int(self.density * ((shape[0] // 2) * (shape[1] // 2)))
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        # Fill borders
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1
        # Make aisles
        for i in range(self.density):
            x, y = randint(0, shape[1]//2 + 1) * 2, randint(0, shape[0]//2 + 1) * 2
            Z[y, x] = 1
            for j in range(self.complexity):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    y_,x_ = neighbours[randint(0, len(neighbours))]
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_
        return Z.astype(int)


class RandomBlockMazeGenerator(MazeGenerator):
    def __init__(self, maze_size, obstacle_ratio):
        super().__init__()
        
        self.maze_size = maze_size
        self.obstacle_ratio = obstacle_ratio
        
        self.maze = self._generate_maze()
        
    def _generate_maze(self):
        maze_size = self.maze_size# - 2  # Without the wall
        
        maze = np.zeros([maze_size, maze_size]) 
        
        # List of all possible locations
        all_idx = np.array(list(cartesian_product(range(maze_size), range(maze_size))))

        # Randomly sample locations according to obstacle_ratio
        random_idx_idx = np.random.choice(maze_size**2, size=int(self.obstacle_ratio*maze_size**2), replace=False)
        random_obs_idx = all_idx[random_idx_idx]

        # Fill obstacles
        for idx in random_obs_idx:
            maze[idx[0], idx[1]] = 1

        # Padding with walls, i.e. ones
        maze = np.pad(maze, 1, 'constant', constant_values=1)
        
        return maze
    

class UMazeGenerator(MazeGenerator):
    def __init__(self, len_long_corridor, len_short_corridor, width, wall_size=1):
        super().__init__()
        
        self.len_long_corridor = len_long_corridor
        self.len_short_corridor = len_short_corridor
        self.width = width
        self.wall_size = wall_size
        
        self.maze = self._generate_maze()
        
    def _generate_maze(self):
        # Make empty maze with appropriate shape
        maze = np.zeros([self.len_short_corridor + 2*self.width, self.len_long_corridor])
        
        # Fill obstables between two long corridors
        maze[self.width : self.width+self.len_short_corridor, : self.len_long_corridor-self.width] = 1
        
        # Padding with walls
        maze = np.pad(maze, self.wall_size, 'constant', constant_values=1)
        
        return maze
    

class TMazeGenerator(MazeGenerator):
    def __init__(self, num_T, T_size, block_size):
        super().__init__()
        
        if num_T%2 == 0:
            raise ValueError('Number of T shape must be odd number, to avoid overlapping of reflected T.')
        self.num_T = num_T
        self.T_size = T_size
        self.block_size = block_size
        
        # Initialize maze with enough space
        height_maze = (block_size[0] + 2) + (num_T*(T_size[0] + 3))
        width_maze = 2*(3*2*T_size[1])
        self.maze = np.ones([height_maze, width_maze])
        
        # Pointer of current index
        self.idx = [-2, width_maze//4]  # initial position to locate start block
        
        # Generate initial start block and initial T shape
        self._generate_start()
        
        # Generate multiple T shapes
        for i in range(num_T - 1):
            if i%2 == 0:  # even number: joint on the left
                self._generate_T(door_direction='left')
            else:  # odd number: joint on the right
                self._generate_T(door_direction='right')
        # Last T shape to the right most
        self._generate_T(door_direction='right')
        
        # Make longer corridor before reflected T shapes to avoid overlapping
        self.maze[self.idx[0], self.idx[1] : self.idx[1]+self.T_size[1]] = 0
        self.idx[1] += (self.T_size[1])  # update index pointer
                
        # Generate multiple reflected T shapes
        for i in range(num_T):
            if i%2 == 0:  # even number: joint on the right
                self._generate_T(door_direction='right', reflected=True)
            else:  # odd number: joint on the left
                self._generate_T(door_direction='left', reflected=True)
                
        # Generate terminal block
        self._generate_end()
        # Clean index pointer
        self.idx = None
        
        # Unpad redundant walls
        self._unpad()

    def _generate_start(self):
        # Fill in start block
        height_start = self.idx[0] - self.block_size[0] + 1
        height_end = self.idx[0] + 1
        width_start = self.idx[1]
        width_end = self.idx[1] + self.block_size[1]
        self.maze[height_start:height_end, width_start:width_end] = 0
        # Update the index pointer at the door, i.e. center of upper edge
        self.idx[0] += (-self.block_size[0])
        self.idx[1] += (self.block_size[1]//2)

        # Fill in initial T shape
        len_corridor, len_arm = self.T_size
        self.maze[self.idx[0]-len_corridor+1 : self.idx[0]+1, self.idx[1]] = 0  # fill in corridor
        # Update index pointer
        self.idx[0] += (-len_corridor)
        # Fill in left and right arms
        self.maze[self.idx[0], self.idx[1]-len_arm : self.idx[1]+len_arm+1] = 0
        # Update index pointer at the right arm
        self.idx[1] += (len_arm + 1)
        
    def _generate_end(self):
        len_corridor, len_arm = self.T_size
        # Fill in the corridor
        self.maze[self.idx[0]-len_corridor//2+1 : -1, self.idx[1]] = 0
        # Update index pointer
        self.idx[0] = -2
        
        # Fill in terminal block
        height, width = self.block_size
        self.maze[-height-1 : -1, self.idx[1]-width//2 : self.idx[1]+width//2+1] = 0
        
    def _generate_T(self, door_direction=None, reflected=False):
        len_corridor, len_arm = self.T_size
        # Fill in corridor
        if reflected:
            offset = [1, 1]
            sign_idx = 1
        else:
            offset = [-1, 0]
            sign_idx = -1
        self.maze[self.idx[0]-len_corridor//2+offset[0] : self.idx[0]+len_corridor//2+offset[1], self.idx[1]] = 0
        # Update index pointer
        self.idx[0] += (sign_idx*len_corridor//2 + offset[1])
        # Fill in left and right arms
        self.maze[self.idx[0], self.idx[1]-len_arm : self.idx[1]+len_arm+1] = 0
        # Update index pointer
        if door_direction == 'left':
            self.idx[1] += (-len_arm - 1)
        elif door_direction == 'right':
            self.idx[1] += (len_arm + 1)
        else:
            raise ValueError('The door direction must be either left or right in string form.')
            
    def _unpad(self):
        """Unpadding for redundant walls"""
        # Unpad top
        while not np.any(self.maze[:2, :] == 0):
            self.maze = self.maze[1:, :]
        # Unpad left
        while not np.any(self.maze[:, :2] == 0):
            self.maze = self.maze[:, 1:]
        # Unpad right
        while not np.any(self.maze[:, -2:] == 0):
            self.maze = self.maze[:, :-1]
            
    def sample_state(self):
        """We want the agent sprawn in start room and goal in the terminal room"""
        height_maze, width_maze = self.maze.shape
        # Free space for possible initial and goal positions
        free_init = self.maze[-self.block_size[0]-1 : -1, :width_maze//2]
        free_goal = self.maze[-self.block_size[0]-1 : -1, width_maze//2:]
        
        # Indices of free spaces
        idxs_init = np.where(free_init == 0)
        idxs_goal = np.where(free_goal == 0)
        
        # Set randomly selected initial and goal positions temporarily
        free_init[np.random.choice(idxs_init[0]), np.random.choice(idxs_init[1])] = 2
        free_goal[np.random.choice(idxs_goal[0]), np.random.choice(idxs_goal[1])] = 3
        
        # Find the global indices for initial and goal position
        idx_init = np.where(self.maze == 2)
        init_state = [idx_init[0][0], idx_init[1][0]]
        idx_goal = np.where(self.maze == 3)
        goal_states = [idx_goal[0][0], idx_goal[1][0]]
        goal_states = [goal_states]  # TODO: multiple goals
        
        # Clean up the initial and goal position in original maze
        self.maze[idx_init[0], idx_init[1]] = 0
        self.maze[idx_goal[0], idx_goal[1]] = 0
        
        return init_state, goal_states
    
    
class WaterMazeGenerator(MazeGenerator):
    def __init__(self, radius_maze, radius_platform):
        super().__init__()
        
        self.radius_maze = radius_maze
        self.radius_platform = radius_platform
        
        # Generate free space for water maze
        self.maze = np.ones([2*self.radius_maze, 2*self.radius_maze])
        self.maze[circle(self.radius_maze, self.radius_maze, self.radius_maze - 1)] = 0
        
        # Generate circular platform
        self.platform = np.zeros_like(self.maze)
        radius_diff = self.radius_maze - self.radius_platform - 1
        valid_x, valid_y = circle(self.radius_maze, self.radius_maze, radius_diff)
        coord_platform = np.stack([valid_x, valid_y], axis=1)[np.random.choice(range(valid_x.shape[0]))]
        self.platform[circle(*coord_platform, self.radius_platform)] = 3
        
        # Add platform to the maze array
        self.maze += self.platform
        
    def sample_state(self):
        """Randomly sample an initial state and goal state within the platform"""
        # Get indices for all free spaces exclude platform, i.e. zero
        free_space = np.where(self.maze + self.platform == 0)
        free_space = list(zip(*free_space))

        # Sample indices for initial state
        init_idx = np.random.choice(len(free_space), size=1)[0]
        
        # Convert initial state to a list, goal states to list of list
        init_state = list(free_space[init_idx])
        
        # Goal states are the states within platform
        goal_states = list(zip(*np.where(self.platform == 3)))
        
        return init_state, goal_states
