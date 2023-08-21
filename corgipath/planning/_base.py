from abc import ABC, abstractmethod
from corgipath.collision._base import BaseCollision
from corgipath.search_space._base import BaseSpace


class BasePlanner(ABC):
    """Abstract class for planning algorithms.

    All planning algorithms must inherit from this class.
    """

    def __init__(self):
        self._collision_system: BaseCollision = None
        self._search_space: BaseSpace = None

    @property
    def collision_system(self):
        return self._collision_system

    @collision_system.setter
    def collision_system(self, collision_system):
        # Check collision system is a subclass of corgipath.collision.BaseCollision
        if not isinstance(collision_system, BaseCollision):
            raise TypeError("Collision system must be a subclass of BaseCollision")
        self._collision_system = collision_system

    @property
    def search_space(self):
        return self._search_space

    @search_space.setter
    def search_space(self, search_space):
        # Check search space is a subclass of corgipath.search_space.BaseSpace
        if not isinstance(search_space, BaseSpace):
            raise TypeError("Search space must be a subclass of BaseSpace")
        self._search_space = search_space
        pass

    @abstractmethod
    def solve(self, start, goal) -> list:
        """Compute a path from start to goal.

        Args:
            start (tuple): Index of start point(or node)
            goal (tuple): Index of goal point(or node)

        Returns:
            list: List of waypoints
        """
        pass
