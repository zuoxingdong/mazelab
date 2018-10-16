from abc import ABC
from abc import abstractmethod

from .base_env import BaseEnv


class BaseNavigationEnv(BaseEnv, ABC):
    r"""Base class for all environment with navigation tasks i.e. agent navigates in a 2D maze to reach goal. 
    
    This can also set to be multi-agents or multi-goals. 
    
    The subclass should implement at least the following:
    
    - :meth:`step`
    - :meth:`reset`
    - :meth:`get_image`
    - :meth:`make_state`
    - :meth:`make_goal`
    - :meth:`is_valid`
    - :meth:`is_goal`
    
    """
    def __init__(self, maze, motion):
        super().__init__(maze, motion)
        
        self.state = self.make_state()
        self.goal = self.make_goal()
    
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
