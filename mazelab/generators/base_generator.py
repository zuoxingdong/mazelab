from abc import ABC
from abc import abstractmethod

from mazelab import ObjectDirectory


class BaseGenerator(ABC):
    r"""Base class for all generators. 
    
    The subclass should implement at least the following:
    
    - :meth:`make_objects`
    - :meth:`__call__`

    """
    def __init__(self):
        self.obj_dir = self.make_objects()
        assert isinstance(self.obj_dir, ObjectDirectory), f'expected as ObjectDirectory, got {type(self.obj_dir)}'
        self.obj_dir = self.obj_dir.objects
        [setattr(self, obj.name, obj) for obj in self.obj_dir.values()]
        
    @abstractmethod
    def make_objects(self):
        r"""Create and return a set of objects with dtype :class:`ObjectDirectory`. """
        pass
        
    @abstractmethod
    def __call__(self):
        r"""Create and return the generated maze in a two dimensional list. """
        pass
