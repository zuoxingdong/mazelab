from dataclasses import dataclass
from dataclasses import field


@dataclass
class Object:
    r"""Defines an object with some of its properties. 
    
    An object can be an obstacle, free space or food etc. It can also have properties like impassable, positions.
    
    """
    name: str
    value: int
    rgb: tuple
    impassable: bool
    positions: list = field(default_factory=list)
