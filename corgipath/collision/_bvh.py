from typing import Tuple, List
import copy
import numpy as np
import collision
from aabbtree import AABB, AABBTree
from ._base import TreeBasedCollision
from .utils import validate_collision_type, Collidable, convert_bounds_to_obstacles


class BoundingVolumeHierarchy(TreeBasedCollision):
    def __init__(self, bounds: Tuple[float, float, float, float]):
        """Bounding volume hierarchy collision system.

        Args:
            bounds (Tuple[float, float, float, float]): The bounds of the environment.
                The format is (x_min, x_max, y_min, y_max).
        """
        super().__init__(bounds)

        self._obstacles: List[Collidable] = convert_bounds_to_obstacles(bounds)

        # Do not handle the tree directly.
        self._aabb_tree = AABBTree()

        # The fallowing members to prevent unnecessary memory allocation.
        self._agent_aabb = AABB([[0.0, 0.0], [0.0, 0.0]])

    def add_obstacles(self, obstacles: list):
        """Add obstacles to the collision system.

        Args:
            obstacles (list): List of obstacles. Each obstacle must be one of the following types:
                `collision.Circle`, `collision.Poly`, `collision.Concave_Poly`

        Raises:
            TypeError: If any of the obstacles is not one of the valid types.
        """
        map(validate_collision_type, obstacles)
        self._obstacles.extend(obstacles)
        self._built = False

    @property
    def obstacles(self) -> List[Collidable]:
        """Get the list of obstacles.

        Note:
            This performs a deep copy of the list.

        Returns:
            List[Union[collision.Circle, collision.Poly, collision.Concave_Poly]]: List of obstacles.
        """
        return copy.deepcopy(self._obstacles)

    @property
    def agent_collision(self) -> Collidable:
        """Get the agent's collision object.

        Note:
            This performs a deep copy of the object.

        Returns:
            Union[collision.Circle, collision.Poly, collision.Concave_Poly]: The agent's collision object.
        """
        # `pos` of `self._agent_collision` used for temporal purposes.
        self._agent_collision.pos = collision.Vector(0.0, 0.0)
        return copy.deepcopy(self._agent_collision)

    @agent_collision.setter
    def agent_collision(self, agent_collision):
        validate_collision_type(agent_collision)
        # The origin of the agent's collision object must be (0, 0).
        if not np.allclose((agent_collision.pos.x, agent_collision.pos.y), (0, 0)):
            raise ValueError("The origin of the agent's collision object must be (0, 0).")
        self._agent_collision = agent_collision

    def build(self):
        """Build the bounding volume hierarchy.

        This method must be called before planning.

        The AABB of objects in the collision package is not a static value,
        causing inefficiencies as it requires recalculations for every request.
        To improve performance, we should consider freezing them into an AABB tree.

        Note:
            This method is time-consuming. Do not call it every time.
            It is recommended to build the collision system only once.
        """
        for obstacle in self._obstacles:
            corners = obstacle.aabb  # ((x_min,y_min), (x_max,y_min), (x_min,y_max), (x_max,y_max))
            limits = ((corners[0][0], corners[3][0]), (corners[0][1], corners[3][1]))  # ((x_min, x_max), (y_min, y_max))
            self._aabb_tree.add(AABB(limits), value=obstacle)
        self._built = True

    def has_collision(self, agent_xyt: Tuple[float, float, float]) -> bool:
        """Check if the agent collides with any obstacle.

        Args:
            agent_xyt (Tuple[float, float, float]): The agent's (x, y, theta) coordinates.

        Returns:
            bool: True if the agent collides with any obstacle, False otherwise.
        """

        # First, check AABB collision
        # ---------------------------
        # Set the agent's position and rotation
        agent_center = collision.Vector(agent_xyt[0], agent_xyt[1])
        self._agent_collision.pos = agent_center
        if not isinstance(self._agent_collision, collision.Circle):
            self._agent_collision.angle = agent_xyt[2]

        # Get the AABB of the agent
        agent_corners = self._agent_collision.aabb
        self._agent_aabb.limits[0][0] = agent_corners[0][0]
        self._agent_aabb.limits[0][1] = agent_corners[3][0]
        self._agent_aabb.limits[1][0] = agent_corners[0][1]
        self._agent_aabb.limits[1][1] = agent_corners[3][1]

        # TODO: compare the performance of the two methods

        # Check if the agent's AABB collides with any obstacle's AABB
        conflict_candidates = self._aabb_tree.overlap_values(self._agent_aabb)

        # Second, check polygon collision
        # -------------------------------
        for obstacle in conflict_candidates:
            if collision.collide(self._agent_collision, obstacle):
                return True

        return False
