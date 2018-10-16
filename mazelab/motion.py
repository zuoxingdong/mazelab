class Motion(object):
    r"""Define the one-step motion. 
    
    Example::
    
        >>> motion = Motion()
        >>> motion.add('north', [-1, 0])
        >>> motion.add('south', [1, 0])
        >>> motion.add('west', [0, -1])
        >>> motion.add('east', [0, 1])
        >>> motion.add('northwest', [-1, -1])
        >>> motion.add('northeast', [-1, 1])
        >>> motion.add('southwest', [1, -1])
        >>> motion.add('southeast', [1, 1])
        >>> motion
        Motion: 
            north: [-1, 0]
            south: [1, 0]
            west: [0, -1]
            east: [0, 1]
            northwest: [-1, -1]
            northeast: [-1, 1]
            southwest: [1, -1]
            southeast: [1, 1]
            
        >>> motion[3]
        ('east', [0, 1])
            
        >>> motion.size
        8
    
    """
    def __init__(self):
        self.motions = []
    
    def add(self, name, delta):
        self.motions.append({name: delta})
        setattr(self, name, delta)
    
    def __getitem__(self, n):
        return list(self.motions[n].items())[0]
    
    @property
    def size(self):
        return len(self.motions)
        
    def __repr__(self):
        string = f'{self.__class__.__name__}: '
        for motion in self.motions:
            key, value = list(motion.items())[0]
            string += f'\n\t{key}: {value}'
            
        return string
