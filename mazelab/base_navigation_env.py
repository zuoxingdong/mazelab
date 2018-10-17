from abc import ABC
from abc import abstractmethod

from copy import deepcopy

from .base_env import BaseEnv


class BaseNavigationEnv(BaseEnv, ABC):
    r"""Base class for all environment with navigation tasks i.e. agent navigates in a 2D maze to reach goal. 
    
    This can also set to be multi-agents or multi-goals. 
    
    The subclass should implement at least the following:
    
    - :meth:`step`
    - :meth:`reset`
    - :meth:`make_state`
    - :meth:`make_goal`
    - :meth:`is_valid`
    - :meth:`is_goal`
    
    """
    def __init__(self, maze, motion):
        super().__init__(maze, motion)
        
        self.state = self.make_state()
        self.goal = self.make_goal()
    
    def get_object_map(self):
        object_map = deepcopy(self.maze.x)
        
        for position in self.state.positions:
            object_map[position[0]][position[1]] = self.state
        
        for position in self.goal.positions:
            object_map[position[0]][position[1]] = self.goal
            
        return object_map
    
    def get_observation(self):
        observation = self.get_object_map()
        for h in range(len(observation)):
            for w in range(len(observation[0])):
                observation[h][w] = observation[h][w].value
                
        return observation
    
    def get_image(self):
        img = self.get_object_map()
        for h in range(len(img)):
            for w in range(len(img[0])):
                img[h][w] = img[h][w].color
                
        return img
    
    @abstractmethod
    def make_state(self):
        pass
    
    @abstractmethod
    def make_goal(self):
        pass
        
    @abstractmethod
    def is_valid(self, position):
        pass
        
    @abstractmethod
    def is_goal(self, position):
        pass
