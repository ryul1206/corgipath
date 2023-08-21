from __future__ import annotations
from collections.abc import Iterable
from typing import Tuple, Dict
from dataclasses import dataclass
import numpy as np
from ._base import BaseGrid


@dataclass
class HybridSuccessor:
    """Successor template for Hybrid A*.

    Attributes:
        xyt (Tuple[float, float, float]): Displacement of X, Y, Theta in local frame.
            X is forward, Y is left, Theta is counter-clockwise.
        cost (float): Total expense to reach this successor. It is usually `edge cost + penalty cost(node cost)`.
    """

    _xyt: Tuple[float, float, float]
    _cost: float

    xyt = property(lambda self: self._xyt)
    cost = property(lambda self: self._cost)

    @classmethod
    def from_heading_with_dist(cls, heading: float, dist: float, cost: float):
        """Create a successor from heading angle and moving distance.

        This method assumes that the agent moves with an **arc** trajectory.
        It is useful when you want to create a successor with a specific moving distance.

        Args:
            heading (float): Heading angle in radian.
            dist (float): Moving distance. `dist > 0.0` is forward, `dist < 0.0` is backward.
            cost (float): Total expense to reach this successor.
        """
        if heading == 0.0:
            return cls((dist, 0.0, 0.0), cost)
        radius = dist / heading
        x = radius * np.sin(heading)
        y = radius * (1.0 - np.cos(heading))
        t = heading
        return cls((x, y, t), cost)

    def apply_to(self, query_node: DefaultHybridNode) -> HybridSuccessor:
        """Apply this successor to a given query node.

        Args:
            query (DefaultHybridNode): Query node in a specific frame X.

        Returns:
            HybridSuccessor: A new successor with the query node applied.
                So the new successor is in the same frame X as the query node.
        """
        query_in_global = np.array(query_node.xyt)
        successor_in_query = np.array(self._xyt)

        # Rotation matrix
        theta = query_in_global[2]
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Successor coordinates in global frame
        xy = R @ successor_in_query[:2] + query_in_global[:2]
        theta = query_in_global[2] + successor_in_query[2]

        # Penalty cost about steering change. This is a trick to improve performance.
        prev_steering_change = query_node.steering_change
        current_steering_change = self._xyt[2]
        """


        # 0 -.


        # """
        # if (current_steering_change - prev_steering_change) >= 0.0:


        return HybridSuccessor((xy[0], xy[1], theta), self._cost)


@dataclass
class DefaultHybridNode:
    xyt: Tuple[float, float, float]
    index: Tuple[int, int, int]
    parent_index: Tuple[int, int, int] = None

    _g_score = np.inf
    _h_score = 0.0
    g_score = property(lambda self: self._g_score)
    h_score = property(lambda self: self._h_score)
    f_score = property(lambda self: self._g_score + self._h_score)

    # Trick to improve performance
    _steering_change = 0.0
    steering_change = property(lambda self: self._steering_change)

    def set_score(self, g_score: float, h_score: float):
        self._g_score = g_score
        self._h_score = h_score


class DefaultHybridGrid(BaseGrid):
    def __init__(self, dxy: float, dtheta: float, node_type=DefaultHybridNode):
        super().__init__(dxy)

        # Validate arguments
        if dtheta <= 0.0:
            raise ValueError("dtheta must be greater than 0.0")
        if not np.isclose((np.pi / dtheta) % 1.0, 0.0):
            print('(WARNING) dtheta must be greater than 0.0')
            # raise ValueError("dtheta must be a divisor of pi")
        if not issubclass(node_type, DefaultHybridNode):
            raise ValueError("node_type must be a subclass of DefaultHybridNode")

        # Configure
        self._dtheta = dtheta
        self._node_type = node_type

        self._successor_template: Tuple[HybridSuccessor] = None

        # Initialize
        self._nodes: Dict[Tuple[int, int, int], DefaultHybridNode] = {}

    @property
    def successor_template(self) -> Tuple[HybridSuccessor]:
        return self._successor_template

    @successor_template.setter
    def successor_template(self, template: Iterable[HybridSuccessor]):
        # Check template is iterable
        if not hasattr(template, "__iter__"):
            raise ValueError("successor_template must be iterable.")
        # Check template is not empty
        if not bool(template):
            raise ValueError("successor_template must not be empty.")
        # Check template is valid
        template = tuple(template)
        map(self._validate_successor, template)
        self._successor_template = template

    def _validate_successor(self, successor: HybridSuccessor):
        # Check if the input is a HybridSuccessor
        if not isinstance(successor, HybridSuccessor):
            raise ValueError("successor must be a HybridSuccessor.")
        # Check if the heading angle is a multiple of dtheta
        if not np.isclose(successor.xyt[2] % self._dtheta, 0.0):
            raise ValueError("The heading angle of a successor state must be a multiple of the angle gap (dtheta) of the grid.")
        # Check if the linear distance exceeds the diagonal distance of grid nodes when heading angle is not changed
        if np.isclose(successor.xyt[2], 0.0):
            dist = np.linalg.norm(successor.xyt[:2])
            diag = np.sqrt(2.0) * self._dxy
            if dist < diag:
                raise ValueError("Linear distance should exceed grid square diagonal when heading not changed.")

    def use_default_successor_template(self, max_heading_change: float, allow_backward: bool = True):
        """Set the successor template to the default template.

        Args:
            max_heading_change (float, optional): The maximum heading change allowed in the template. The unit is radian.
            allow_backward (bool, optional): Whether to allow backward motion in the template. Defaults to True.
        """
        # In Hybrid A*, minimum distance to forward >= diagonal length.
        diagonal = np.sqrt(2.0) * self._dxy
        # Pair of displacement and cost
        backward_penalty = 2.0
        forward = (diagonal, diagonal)
        backward = (-diagonal, diagonal * backward_penalty)
        edges = (forward, backward) if allow_backward else (forward,)
        # Make template
        template = []
        for displacement, cost in edges:
            template.append(HybridSuccessor.from_heading_with_dist(0.0, displacement, cost))
            _dist = displacement / np.sqrt(2.0)
            _cost = cost / np.sqrt(2.0)
            for i in range(1, int(max_heading_change / self._dtheta) + 1):
                template.append(HybridSuccessor.from_heading_with_dist(self._dtheta * i, _dist, _cost))
                template.append(HybridSuccessor.from_heading_with_dist(-self._dtheta * i, _dist, _cost))
        self.successor_template = template      

    def get_successors_of(self, query_node: DefaultHybridNode) -> Iterable[HybridSuccessor]:
        for successor in self._successor_template:
            yield successor.apply_to(query_node)

    def reset(self):
        self._nodes.clear()

    def is_prepared(self) -> bool:
        is_empty = not bool(self._nodes)
        is_template = self._successor_template is not None
        return is_empty and is_template

    def grid_index(self, xyt: Tuple[float, float, float]) -> Tuple[int, int, int]:
        x, y, theta = xyt
        i = int(np.rint(x / self._dxy))
        j = int(np.rint(y / self._dxy))
        k = int(np.rint((theta % np.pi) / self._dtheta))
        return (i, j, k)

    def get_node(self, xyt: Tuple[float, float, float]) -> DefaultHybridNode:
        index = self.grid_index(xyt)
        if index not in self._nodes:
            self._nodes[index] = self._node_type(xyt, index)
        return index, self._nodes[index]

    def try_get_node_with_index(self, index: Tuple[int, int, int]) -> DefaultHybridNode:
        return self._nodes[index]
