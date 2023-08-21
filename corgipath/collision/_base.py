from abc import ABC, abstractmethod
from typing import Tuple
import copy
import numpy as np
import collision
from .utils import validate_collision_type, Collidable


class BaseCollision(ABC):
    """Abstract class for collision systems.

    All collision systems must inherit from this class.
    Fallowing methods must be implemented:
        - is_prepared()
    """

    def __init__(self):
        pass

    @abstractmethod
    def is_prepared(self) -> bool:
        """Implement this method to check if the collision system is ready for planning.

        Returns:
            bool: True if prepared, False otherwise
        """
        pass


class TreeBasedCollision(BaseCollision):
    """Abstract class for tree-based collision systems.

    Fallowing methods must be implemented:
        - build()
        - has_collision()
    """

    def __init__(self, bounds: Tuple[float, float, float, float]):
        """Initialize the tree-based collision system.

        Args:
            bounds (Tuple[float, float, float, float]): The bounds of the environment.
                The format is (x_min, x_max, y_min, y_max).
        """
        super().__init__()

        if len(bounds) != 4:
            raise ValueError("bounds must be a tuple of length 4. (xmin, xmax, ymin, ymax)")
        if bounds[0] >= bounds[1]:
            raise ValueError("bounds: xmin must be less than xmax.")
        if bounds[2] >= bounds[3]:
            raise ValueError("bounds: ymin must be less than ymax.")

        self._bounds = bounds
        self._built = False
        self._agent_collision = None

    def is_prepared(self) -> bool:
        """Check if the collision system is ready for planning.

        To be ready, the collision system must be built and the agent's collision object must be set.
            - To build the collision system, call `build()`.
            - To set the agent's collision object, set `agent_collision`.

        Returns:
            bool: True if prepared, False otherwise
        """
        return self._built and (self._agent_collision is not None)

    @abstractmethod
    def build(self):
        """Implement this method to build the tree structure.

        Building the tree structure is time-consuming. Do not call it every time.
        It is recommended to build the collision system only once.

        Note:
            `self._built` must be set to True at the end of this method.
            And `self._built` must be set to False when any obstacle is added or removed.
        """
        pass

    @abstractmethod
    def has_collision(self) -> bool:
        """Implement this method to check if the agent collides with any obstacle."""
        pass
