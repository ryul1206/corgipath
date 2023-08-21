from typing import Tuple, List
import copy
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
import collision
from ._base import TreeBasedCollision
from . import utils


class KDTree(TreeBasedCollision):
    def __init__(self, bounds: Tuple[float, float, float, float], min_resolution: float):
        """KDTree collision system.

        Args:
            bounds (Tuple[float, float, float, float]): The bounds of the environment.
                The format is (x_min, x_max, y_min, y_max).
        """
        super().__init__(bounds)
        self._min_resolution = min_resolution

        # List of obstacles. List of (x, y) points.
        _boundary_vertices = [
            np.array([bounds[0], bounds[2]]),
            np.array([bounds[1], bounds[2]]),
            np.array([bounds[1], bounds[3]]),
            np.array([bounds[0], bounds[3]]),
        ]
        _boundary_points = [
            utils.convert_line_to_points(_boundary_vertices[i], _boundary_vertices[(i + 1) % 4], self._min_resolution)
            for i in range(4)
        ]
        self._obstacles: np.ndarray = np.concatenate(_boundary_points)

        # Do not handle the tree directly.
        self._kd_tree = None

        self._built = False
        self._agent_collision = None
        self._is_agent_circle = False
        self._window_radius = 0.0  # Collision window to be used in `self._kd_tree.query_ball_point`.

    def add_obstacles(self, obstacles: list):
        """Add obstacles to the collision system.

        Args:
            obstacles (list): List of obstacles. Each obstacle must be one of the following types:
                - (x, y) point type: `Tuple[float, float]`, `List[float, float]`, `np.ndarray`, `collision.Vector`
                - Collision object type: `collision.Circle`, `collision.Poly`, `collision.Concave_Poly`

        Raises:
            TypeError: If any of the obstacles is not one of the valid types.
        """
        _xy = []
        for obstacle in obstacles:
            if isinstance(obstacle, (collision.Circle, collision.Poly, collision.Concave_Poly)):
                _outline = utils.extract_outline_as_points(obstacle, self._min_resolution)
                _xy.extend(_outline.tolist())
            elif isinstance(obstacle, collision.Vector):
                _xy.append((obstacle.x, obstacle.y))
            elif isinstance(obstacle, (tuple, list, np.ndarray)):
                if len(obstacle) != 2:
                    raise ValueError(f"The obstacle must be a 2D point. Got {obstacle} (size: {len(obstacle)}) instead.")
                _xy.append(obstacle)
            else:
                raise TypeError(f"Invalid obstacle type. Got {type(obstacle)} instead.")
        # Concatenate the new obstacles to the existing ones.
        self._obstacles = np.concatenate((self._obstacles, np.array(_xy)))
        self._built = False

    @property
    def obstacles(self) -> np.ndarray:
        """Get the list of obstacles.

        Note:
            This performs a deep copy of the list.

        Returns:
            np.ndarray: List of obstacles.
        """
        return copy.deepcopy(self._obstacles)

    @property
    def agent_collision(self) -> utils.Collidable:
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
        """Set the agent's collision object.

        Collision type must be one of the following:
            - `collision.Circle` (recommended): Most efficient.
            - `collision.Poly`
            - `collision.Concave_Poly`

        Args:
            agent_collision (_type_): _description_

        Raises:
            ValueError: _description_
        """
        utils.validate_collision_type(agent_collision)
        # The origin of the agent's collision object must be (0, 0).
        if not np.allclose((agent_collision.pos.x, agent_collision.pos.y), (0, 0)):
            raise ValueError("The origin of the agent's collision object must be (0, 0).")
        self._agent_collision = agent_collision
        # Update collision window radius.
        if isinstance(agent_collision, collision.Circle):
            self._is_agent_circle = True
            self._window_radius = agent_collision.radius
        else:
            self._is_agent_circle = False
            # Get the maximum distance from the origin.
            for vertex in agent_collision.points:
                dist = np.sqrt(vertex.x**2 + vertex.y**2)
                if dist > self._window_radius:
                    self._window_radius = dist

    def build(self):
        """Build the KDTree.

        This method must be called before planning.
        """
        self._kd_tree = cKDTree(self._obstacles)
        self._built = True

    def has_collision(self, agent_xyt: Tuple[float, float, float]) -> bool:
        """Check if the agent collides with any obstacles.

        Args:
            agent_xyt (Tuple[float, float, float]): The agent state.
                The format is (x, y, theta).

        Returns:
            bool: True if the agent collides with any obstacles.
        """
        if self._is_agent_circle:
            # Get the closest obstacle.
            closest_dist, _ = self._kd_tree.query(agent_xyt[:2])
            return closest_dist < self._window_radius

        # collision.Poly and collision.Concave_Poly
        self._agent_collision.pos = collision.Vector(agent_xyt[0], agent_xyt[1])
        self._agent_collision.angle = agent_xyt[2]
        agent_vertices = [(v.x, v.y) for v in self._agent_collision.points]
        polygon = Polygon(agent_vertices)

        # Get the obstacles within the collision window.
        points: list = self._kd_tree.query_ball_point(agent_xyt[:2], self._window_radius)
        for point in points:
            # Checks if the polygon contains the point.
            if polygon.contains(Point(self._obstacles[point])):
                return True
        return False
