from typing import List, Tuple, Union
import copy
import numpy as np
import matplotlib.pyplot as plt
import collision
from corgipath.collision.utils import Collidable
from corgipath.search_space import BaseGrid


def draw_coordinates(ax: plt.Axes, xyt: Tuple[float, float, float], style: dict = {}):
    """Draw coordinate system at the given position and orientation.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw.
        xyt (Tuple[float, float, float]): A tuple of (x, y, theta). Theta is in radian.
        style (dict, optional): Style options for drawing. Defaults to empty dict.

            - `coordinates_size`: The length from origin to the end of the forward vector. Defaults to 1.0.
            - `coordinates_type`: Type of coordinates. Defaults to "arrow".
                Available types: `arrow`, `directional circle`
            - The keyward arguments from matplotlib.pathes.Patch are also available. See below for details.
                Url: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html

    Raises:
        AttributeError: Unexpected keyword arguments in `style`.
        ValueError: Unknown coordinates type.
    """
    style = style.copy()
    coordinates_size = style.pop("coordinates_size", 1.0)
    coordinates_type = style.pop("coordinates_type", "arrow")

    sx, sy, theta = xyt
    dx, dy = coordinates_size * np.cos(theta), coordinates_size * np.sin(theta)

    if coordinates_type == "arrow":
        ax.arrow(sx, sy, dx, dy, **style)
    elif coordinates_type == "directional circle":
        # Draw a circle for origin
        circle = plt.Circle((sx, sy), coordinates_size / 4.0, **style)
        ax.add_artist(circle)
        # Draw a line for direction
        if "fill" in style:
            style.pop("fill")  # Style for line. Remove fill
        line = plt.Line2D((sx, sx + dx), (sy, sy + dy), **style)
        ax.add_artist(line)
    else:
        raise ValueError(f"Unknown coordinates type: {coordinates_type}")


def draw_shape(ax: plt.Axes, shape: Collidable, at: Tuple[float, float, float], style: dict = {}):
    """Draw outer shape of a collidable object.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw.
        shape (Collidable): A collidable object.
            This method only takes into account the outer shape of the object in its local frame,
            disregarding the object's position and orientation.
        at (Tuple[float, float, float]): The (x, y, theta) coordinates of the shape in the world frame.
        style (dict, optional): Style options for drawing. Defaults to empty dict.
            See matplotlib.pathes.Patch for details.
            Url: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html
    """
    if isinstance(shape, collision.Circle):
        center_xy = (at[0], at[1])
        circle = plt.Circle(center_xy, shape.radius, **style)
        ax.add_artist(circle)
    elif isinstance(shape, (collision.Poly, collision.Concave_Poly)):
        # Get a copy of the shape
        temp = copy.deepcopy(shape)
        temp.pos = collision.Vector(at[0], at[1])
        temp.angle = at[2]
        # Get the outer shape of temp
        points: List[collision.Vector] = temp.points
        polygon = plt.Polygon([(p.x, p.y) for p in points], **style)
        ax.add_artist(polygon)


def draw_collsion_objects(ax: plt.Axes, collsion_objects: List[Collidable], style: dict = {}):
    """Draw collision objects.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw.
        collsion_objects (List[Collidable]): List of collision objects.
        style (dict, optional): Style options for drawing. Defaults to empty dict.
            See matplotlib.pathes.Patch for details.
            Url: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html
    """
    for obstacle in collsion_objects:
        theta = obstacle.angle if hasattr(obstacle, "angle") else 0.0
        draw_shape(ax, obstacle, (obstacle.pos.x, obstacle.pos.y, theta), style)


def draw_dots(ax: plt.Axes, dots: Union[List[Tuple[float, float]], np.ndarray], style: dict = {}):
    """Draw dots.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw.
        dots (Union[List[Tuple[float, float]], np.ndarray]): List of (x, y) coordinates of the dots.
            If `dots` is a numpy array, it should be of shape (N, 2).
        style (dict, optional): Style options for drawing. Defaults to empty dict.
            See matplotlib.pathes.Patch for details.
            Url: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html
    """
    for dot in dots:
        ax.plot(dot[0], dot[1], **style)


def draw_waypoints(
    ax: plt.Axes,
    waypoints: List[Tuple[float, float, float]],
    shape: Collidable,
    show_shape: bool = True,
    shape_style: dict = {},
    show_coordinates: bool = False,
    coordinates_style: dict = {},
):
    """Draw waypoints.

    Note:
        The `shape_style` and `coordinates_style` are passed to `draw_shape` and `draw_coordinates` respectively.
        See the documentation of these methods for details.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw.
        waypoints (List[Tuple[float, float, float]]): A list of the (x, y, theta) of the shape in the world frame.
        shape (Collidable): A collidable object.
            This method only takes into account the outer shape of the object in its local frame,
            disregarding the object's position and orientation.
        show_shape (bool): Draw shape only when it is True. Defaults to True.
        shape_style (dict, optional): Style options for drawing the shape. Defaults to empty dict.
            See `draw_shape` for details.
        show_coordinates (bool): Draw coordinates only when it is True. Defaults to False.
        coordinates_style (dict, optional): Style options for drawing the coordinates. Defaults to empty dict.
            See `draw_coordinates` for details.
    """
    for xyt in waypoints:
        if show_shape:
            draw_shape(ax, shape, xyt, shape_style)
        if show_coordinates:
            draw_coordinates(ax, xyt, coordinates_style)


def draw_grid(ax: plt.Axes, grid: BaseGrid, drawing_bounds: Tuple[float, float, float, float], style: dict = {}):
    """Draw grid.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw.
        grid (BaseGrid): A grid object.
        drawing_bounds (Tuple[float, float, float, float]): The (xmin, xmax, ymin, ymax) of the drawing area.
        style (dict, optional): Style options for drawing. Defaults to empty dict.
            See matplotlib.pathes.Patch for details.
            Url: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html
    """
    grid_size: float = grid.dxy
    xmin, xmax, ymin, ymax = drawing_bounds
    # Draw vertical lines
    xs = np.arange(grid_size / 2.0, xmax, grid_size)
    xs = np.append(xs, np.arange(-grid_size / 2.0, xmin, -grid_size))
    [ax.add_artist(plt.Line2D((x, x), (ymin, ymax), **style)) for x in xs]
    # Draw horizontal lines
    ys = np.arange(grid_size / 2.0, ymax, grid_size)
    ys = np.append(ys, np.arange(-grid_size / 2.0, ymin, -grid_size))
    [ax.add_artist(plt.Line2D((xmin, xmax), (y, y), **style)) for y in ys]
