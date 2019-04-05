import numpy as np

from skimage.draw import rectangle


def t_maze(t_shape, thick):
    assert t_shape[0] % 2 == 0, 'top bar size should be even number for symmetric shape. '
    x = np.ones([t_shape[1] + thick + 2, t_shape[0] + thick + 2], dtype=np.uint8)
    
    rr, cc = rectangle([1, 1], extent=[thick, t_shape[0] + thick])
    x[rr, cc] = 0
    
    rr, cc = rectangle([1 + thick, x.shape[1]//2 - thick//2], extent=[t_shape[1], thick])
    x[rr, cc] = 0
    
    return x
