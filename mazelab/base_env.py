from abc import ABC
from abc import abstractmethod

import numpy as np

import gym
from gym.utils import seeding
from gym import spaces

from PIL import Image


class BaseEnv(gym.Env, ABC):
    """Base class for all mazelab environments. 
    
    The subclass should implement at least the following:
    
    - :meth:`step`
    - :meth:`reset`
    - :meth:`get_image`

    """
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second' : 3}
    
    def __init__(self, maze, motion):
        self.maze = maze
        self.motion = motion
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.maze.size, dtype=np.float32)
        self.action_space = spaces.Discrete(self.motion.size)
        
        self.viewer = None
        
        self.seed()
    
    @abstractmethod
    def step(self, action):
        pass
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def get_image(self):
        pass
    
    def render(self, mode='human', max_width=500):
        img = self.get_image()
        img = np.asarray(img).astype(np.uint8)
        img_height, img_width = img.shape[:2]
        ratio = max_width/img_width
        img = Image.fromarray(img).resize([int(ratio*img_width), int(ratio*img_height)])
        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control.rendering import SimpleImageViewer
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            
            return self.viewer.isopen
            
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
