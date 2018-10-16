from dataclasses import dataclass
from dataclasses import field


@dataclass
class Object:
    r"""Defines an object with some of its properties. 
    
    An object can be an obstacle, free space or food etc. It can also have properties like impassable, positions etc.
    
    Example::
    
        >>> obj = Object('free', 0, 'white', True)
        >>> obj
        Object(name='free', value=0, color='white', impassable=True, positions=[])
    
    """
    name: str
    value: int
    color: str
    impassable: bool
    positions: list = field(default_factory=list)


class ObjectDirectory(object):
    r"""A directory of name-value pairs for all necessary objects. 
    
    It maintains a dictionary of objects where the key is the name of the object. 
    
    If some of object attributes has conflict with existing ones, then an AssertionError will be raised.
    
    Example::
    
        >>> objects = ObjectDirectory(redundancy_checklist=[])
        >>> objects.add_object('obstacle', 1, 'black', True, [[1, 2]])
        >>> objects.add_object('free', 0, 'white', False, [[2, 2]])
        >>> objects
        ObjectDirectory: 
            Object(name='obstacle', value=1, color='black', impassable=True, positions=[[1, 2]])
            Object(name='free', value=0, color='white', impassable=False, positions=[[2, 2]])
    
    """
    def __init__(self, redundancy_checklist=[]):
        self.objects = {}
        self.redundancy_checklist = redundancy_checklist
    
    def add(self, obj):
        r"""Add an object to the directory. 
        
        Args:
            obj (Object): an object
        """
        self._check(obj)
        
        self.objects[obj.name] = obj
        setattr(self, obj.name, obj)
    
    def add_object(self, name, value, color, impassable, positions=[]):
        obj = Object(name, value, color, impassable, positions)
        self.add(obj)
        
    def _check(self, obj):
        assert isinstance(obj, Object)
        
        assert not hasattr(self, obj.name), f'object with name {obj.name} already existed'
        
        for item in self.redundancy_checklist:
            all_item = [getattr(exist_obj, item) for exist_obj in self.objects.values()]
            assert getattr(obj, item) not in all_item, f'{item} {getattr(obj, item)} already existed'
        
    def __repr__(self):
        string = f'{self.__class__.__name__}: '
        for obj in self.objects.values():
            string += f'\n\t{obj}'
        
        return string
