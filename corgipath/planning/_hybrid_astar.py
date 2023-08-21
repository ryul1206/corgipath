from typing import Tuple, List, Dict
import heapq
import inspect
import numpy as np
from ._base import BasePlanner
from corgipath.search_space import DefaultHybridNode
from corgipath.matplot.live_draw import LiveDrawOption

import time


def default_heuristic(query: DefaultHybridNode, goal: DefaultHybridNode) -> float:
    """Default heuristic function for Hybrid A*.

    This heuristic function assumes Z-shaped path to approximate a cubic Bézier curve.

    Args:
        query (DefaultHybridNode): Query node. It should be subclass of DefaultHybridNode.
        goal (DefaultHybridNode): Goal node. It should be subclass of DefaultHybridNode.

    Returns:
        float: Heuristic score from current node to goal
    """
    qx, qy, qrad = query.xyt
    gx, gy, grad = goal.xyt
    qxy = np.array((qx, qy))
    gxy = np.array((gx, gy))
    dist = np.linalg.norm(gxy - qxy)
    # Offset for intermediary points
    ratio = 0.5
    offset_length = ratio * dist
    # Two intermediary point A and B
    axy = qxy + offset_length * np.array((np.cos(qrad), np.sin(qrad)))
    bxy = gxy - offset_length * np.array((np.cos(grad), np.sin(grad)))
    # Heuristic score
    h_score = np.linalg.norm(gxy - bxy) + np.linalg.norm(bxy - axy) + np.linalg.norm(axy - qxy)
    return h_score


def default_terminal_condition(query: DefaultHybridNode, goal: DefaultHybridNode) -> bool:
    """Default terminal condition for Hybrid A*.

    Args:
        query (DefaultHybridNode): Query node. It should be subclass of DefaultHybridNode.
        goal (DefaultHybridNode): Goal node. It should be subclass of DefaultHybridNode.

    Returns:
        bool: True if the query index is the same as the goal index.
    """
    return query.index == goal.index


class HybridAstar(BasePlanner):
    def __init__(self):
        super().__init__()

        # Live draw
        self._live_draw = {
            "focus_current_node": LiveDrawOption(),
            "defocus_current_node": LiveDrawOption(),
            "open_list": LiveDrawOption(),
            "path_reconstruction": LiveDrawOption(),
        }
        self._live_enabled = dict.fromkeys(self._live_draw.keys(), False)

    def _validate_live_draw_option(self, option: LiveDrawOption):
        # Option
        if not isinstance(option, LiveDrawOption):
            raise TypeError(f"Option must be an instance of LiveDrawOption. Got {type(option)}")
        # Draw function
        if option.draw_func is None:
            return
        if not callable(option.draw_func):
            raise ValueError("`draw_func` must be a callable")
        sig = inspect.signature(option.draw_func)
        if len(sig.parameters) != 1:
            raise ValueError("`draw_func` must have only one input argument, which is a tuple of (x, y, theta)")

    def set_live_draw_options(self, options: Dict[str, LiveDrawOption]):
        """Set options for live draw.

        Args:
            options (Dict[str, LiveDrawOption]): Live draw options
                The input argument of `draw_func` must be a tuple of (x, y, theta).
        """
        for key, option in options.items():
            # Validate key
            if key not in self._live_draw:
                raise ValueError(f"Invalid key: {key}\nValid keys: {self._live_draw.keys()}")
            # Validate option
            self._validate_live_draw_option(option)
            self._live_enabled[key] = False if option.draw_func is None else True
            self._live_draw[key] = option

    def _reconstruct_path(self, current_node: DefaultHybridNode) -> List[Tuple[float, float, float]]:
        """Reconstruct the path from start to goal.

        Args:
            current_node (DefaultHybridNode): Current node. Usually it is the goal node.

        Returns:
            List[Tuple[float, float, float]]: Waypoints (List of X, Y, Theta)
        """
        # Get the path from goal to start
        inverse_path = [current_node]
        _n = current_node

        if self._live_enabled["path_reconstruction"]:
            self._live_draw["path_reconstruction"].draw(_n.xyt)

        while _n.parent_index is not None:
            _n = self._search_space.try_get_node_with_index(_n.parent_index)
            inverse_path.append(_n)

            if self._live_enabled["path_reconstruction"]:
                self._live_draw["path_reconstruction"].draw(_n.xyt)

        # Reverse the path. Now it is from start to goal
        path = [n.xyt for n in reversed(inverse_path)]
        return path

    def solve(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        fn_heuristic=default_heuristic,
        fn_terminal_condition=default_terminal_condition,
    ) -> List[Tuple[float, float, float]]:
        """Solve the planning problem.

        Args:
            start (Tuple[float, float, float]): Start X, Y, Theta
            goal (Tuple[float, float, float]): Goal X, Y, Theta
            fn_heuristic (Callable, optional): Heuristic function. Defaults to `default_heuristic`.
            fn_terminal_condition (Callable, optional): Terminal condition function. Defaults to `default_terminal_condition`.

        Raises:
            RuntimeError: If collision system and search space are not prepared

        Returns:
            List[Tuple[float, float, float]]: Waypoints (List of X, Y, Theta). Empty list if no solution is found.
        """
        if (not self._collision_system.is_prepared()) or (not self._search_space.is_prepared()):
            raise RuntimeError("Collision system and search space must be prepared before planning")

        # Prepare for planning
        grid = self._search_space
        env = self._collision_system

        # Algorithm begins here
        start_index, start_node = grid.get_node(start)
        goal_index, goal_node = grid.get_node(goal)
        start_node.set_score(g_score=0.0, h_score=fn_heuristic(start_node, goal_node))
        print("\n---------------------------------")
        print(f"Start node: {start_node}, Start index: {start_index}")
        print(f"Goal node: {goal_node}, Goal index: {goal_index}")
        print("---------------------------------")
        print("Start planning...")

        # Validate start and goal
        if env.has_collision(start):
            raise RuntimeError(f"Start node {start} is in collision")
        if env.has_collision(goal):
            raise RuntimeError(f"Goal node {goal} is in collision")

        # Open heap: (f_score, node_index)
        open_heap = [(start_node.f_score, start_index)]

        # Logs
        _count_node_expanded = 0

        start_time = time.time()

        while open_heap:
            if time.time() - start_time > 3:
                print("Time out!")
                return []
            current_index: Tuple[int, int, int] = heapq.heappop(open_heap)[1]
            current_node: DefaultHybridNode = grid.try_get_node_with_index(current_index)
            _count_node_expanded += 1

            if self._live_enabled["focus_current_node"]:
                self._live_draw["focus_current_node"].draw(current_node.xyt)

            # Comment out this line to fix the passover bug
            # if fn_terminal_condition(current_node, goal_node):
            #     # Solution found
            #     print(f"Goal found! {current_node.xyt} (Input goal: {goal}))")
            #     print(f"Nodes expanded: {_count_node_expanded}")
            #     print(f"Time elapsed: {time.time() - start_time:.3f} seconds\n")
            #     return self._reconstruct_path(current_node)

            for s in grid.get_successors_of(current_node):
                # Check collision
                if env.has_collision(s.xyt):
                    continue

                # Update node
                neighbor_index, neighbor_node = grid.get_node(s.xyt)

                # print(f"Neighbor node: {neighbor_node}, Neighbor index: {neighbor_index}")
                tentative_g_score = current_node.g_score + s.cost
                if tentative_g_score < neighbor_node.g_score:
                    neighbor_node.set_score(g_score=tentative_g_score, h_score=fn_heuristic(neighbor_node, goal_node))
                    neighbor_node.xyt = s.xyt
                    neighbor_node.parent_index = current_index

                    # Ignore the old (f_score, index) in the heap, even it still exists in the heap.
                    # Re-sorting and removing the old data is time consuming. Just add the new one.
                    heapq.heappush(open_heap, (neighbor_node.f_score, neighbor_index))
                    # 0.2~0.3 secd

                    if self._live_enabled["open_list"]:
                        self._live_draw["open_list"].draw(neighbor_node.xyt)

                if fn_terminal_condition(neighbor_node, goal_node):
                    # Solution found
                    print(f"Goal found! {neighbor_node.xyt} (Input goal: {goal}))")
                    print(f"Nodes expanded: {_count_node_expanded}")
                    print(f"Time elapsed: {time.time() - start_time:.3f} seconds (Visualization time is {'' if True in self._live_enabled.values() else 'not '}included)\n")
                    return self._reconstruct_path(neighbor_node)

                """ 10~15 secs
                # Update node
                tentative_g_score = current_node.g_score + s.cost
                neighbor_node = grid.get_node(s.xyt)
                if tentative_g_score < neighbor_node.g_score:
                    neighbor_node.set_score(g_score=tentative_g_score, h_score=fn_heuristic(s, goal_node))
                    neighbor_node.xyt = s.xyt
                    neighbor_node.parent_index = current_index

                # Check if neighbor is in open heap
                if neighbor_node in open_heap:
                    # Sorting is time consuming. Remove and re-add is faster
                    open_heap.remove(neighbor_node)
                heapq.heappush(open_heap, neighbor_node)
                """

            if self._live_enabled["defocus_current_node"]:
                self._live_draw["defocus_current_node"].draw(current_node.xyt)

        # No solution found
        return []
