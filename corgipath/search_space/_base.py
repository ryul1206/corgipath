from abc import ABC, abstractmethod


# class BaseNode(ABC):
#     """Abstract class for nodes.

#     All nodes must inherit from this class.
#     """

#     def __init__(self):
#         pass


class BaseSpace(ABC):
    """Abstract class for search spaces.

    All search spaces must inherit from this class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def is_prepared(self) -> bool:
        """Check if the search space is prepared.

        Returns:
            bool: True if prepared, False otherwise
        """
        return False


class BaseGrid(BaseSpace):
    """Abstract class for grids.

    All grids must inherit from this class.
    """

    def __init__(self, dxy: float):
        super().__init__()

        # Validate arguments
        if dxy <= 0.0:
            raise ValueError("dxy must be greater than 0.0")

        # Configure
        self._dxy = dxy

    @abstractmethod
    def is_prepared(self) -> bool:
        """Check if the grid is prepared.

        Returns:
            bool: True if prepared, False otherwise
        """
        return False

    @property
    def dxy(self) -> float:
        """Delta x and y in the grid. (Grid resolution)"""
        return self._dxy
