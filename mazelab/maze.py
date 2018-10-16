import numpy as np


class Maze(object):
    def __init__(self, generator):
        self.generator = generator
        
        self.x = self.generator()
        
    @property
    def size(self):
        return len(self.x), len(self.x[0])
    
    def _to_item(self, item):
        H, W = self.size
        x = np.zeros([H, W]).tolist()
        for h in range(H):
            for w in range(W):
                x[h][w] = getattr(self.x[h][w], item)
                
        return x
    
    def to_name(self):
        return self._to_item('name')
    
    def to_value(self):
        return np.asarray(self._to_item('value'))
    
    def to_color(self):
        return self._to_item('color')
    
    def to_impassable(self):
        return self._to_item('impassable')
    
    @property
    def free_space(self):
        x = self.to_value()
        idx = np.where(x == self.generator.free.value)
        idx = np.dstack(idx)[0]
        
        return idx
