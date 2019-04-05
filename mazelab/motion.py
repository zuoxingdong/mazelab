from collections import namedtuple


VonNeumannMotion = namedtuple('VonNeumannMotion', 
                              ['north', 'south', 'west', 'east'], 
                              defaults=[[-1, 0], [1, 0], [0, -1], [0, 1]])


MooreMotion = namedtuple('MooreMotion', 
                         ['north', 'south', 'west', 'east', 
                          'northwest', 'northeast', 'southwest', 'southeast'], 
                         defaults=[[-1, 0], [1, 0], [0, -1], [0, 1], 
                                   [-1, -1], [-1, 1], [1, -1], [1, 1]])
