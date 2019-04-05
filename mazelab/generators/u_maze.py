import numpy as np

from skimage.draw import rectangle


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
