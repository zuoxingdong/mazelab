import numpy as np

from skimage.draw import rectangle
from skimage.draw import circle
from skimage.draw import random_shapes


def random_maze(width=81, height=51, complexity=.75, density=.75):
    r"""Generate a random maze array. 
    
    It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
    is ``1`` and for free space is ``0``. 
    
    Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
    """
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = np.random.randint(0, shape[1]//2 + 1) * 2, np.random.randint(0, shape[0]//2 + 1) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_,x_ = neighbours[np.random.randint(0, len(neighbours))]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
                    
    return Z.astype(int)


def u_maze(width, height, obstacle_width, obstacle_height):
    x = np.zeros([height, width], dtype=np.uint8)
    # wall
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1
    
    start = [height//2 - obstacle_height//2, 1]
    rr, cc = rectangle(start, extent=[obstacle_height, obstacle_width], shape=x.shape)
    x[rr, cc] = 1
    
    return x


def morris_water_maze(radius, platform_center, platform_radius):
    x = np.ones([2*radius, 2*radius], dtype=np.uint8)
    
    rr, cc = circle(radius, radius, radius - 1)
    x[rr, cc] = 0
    
    platform = np.zeros_like(x)
    rr, cc = circle(*platform_center, platform_radius)
    platform[rr, cc] = 2
    x += platform
    
    return x


def random_shape_maze(width, height, max_shapes, max_size, allow_overlap, shape=None):
    x, _ = random_shapes([height, width], max_shapes, max_size=max_size, multichannel=False, shape=shape, allow_overlap=allow_overlap)
    
    x[x == 255] = 0
    x[np.nonzero(x)] = 1
    
    # wall
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1
    
    return x


def t_maze(t_shape, thick):
    assert t_shape[0] % 2 == 0, 'top bar size should be even number for symmetric shape. '
    x = np.ones([t_shape[1] + thick + 2, t_shape[0] + thick + 2], dtype=np.uint8)
    
    rr, cc = rectangle([1, 1], extent=[thick, t_shape[0] + thick])
    x[rr, cc] = 0
    
    rr, cc = rectangle([1 + thick, x.shape[1]//2 - thick//2], extent=[t_shape[1], thick])
    x[rr, cc] = 0
    
    return x


def double_t_maze():
    x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], 
                  [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
                  [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
                  [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
                  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], 
                  [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
                  [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
                  [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
    
    return x
