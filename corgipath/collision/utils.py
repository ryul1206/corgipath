from typing import List, Tuple, Union
from PIL import Image
import numpy as np
import collision


# Type alias(es)
Collidable = Union[collision.Circle, collision.Poly, collision.Concave_Poly]


def get_threshold_pixels_uv(image_path: str, threshold: int):
    """Get the UV coordinates of the pixels that have passed the threshold.

    Args:
        image_path (str): Path to the image.
        threshold (int): Threshold value. Pixels with values below this threshold will be returned.

    Returns:
        np.ndarray: Array of UV coordinates of the pixels that have passed the threshold.
    """
    # Open the image using PIL
    img = Image.open(image_path)

    # Convert the image to grayscale
    img = img.convert("L")

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Find the coordinates of the pixels that have passed the threshold
    pixels = np.argwhere(img_array < threshold)

    # Reverse order of the coordinates from (v,u) to (u,v)
    pixels = pixels[:, [1, 0]]
    return pixels


def get_obstacles_from_image(
    image_path: str, threshold: int, pixel_size: float, origin_in_uv: Tuple[float, float]
) -> List[collision.Circle]:
    """Get the obstacles from an occupancy grid image.

    Args:
        image_path (str): Path to the image.
        threshold (int): Threshold value. Pixels with values below this threshold are considered obstacles.
        pixel_size (float): Represents the width in meters of a single pixel.
        origin_in_uv (tuple[float, float]): The origin of the image in UV coordinates. Each value is in the range [0, 1].
            For example, if the origin is located at the bottom left corner of the image, the value would be (0, 1).
            When the origin is in the center, the value is (0.5, 0.5).

    Returns:
        list[collision.Circle]: List of obstacles. Each obstacle is a circle with a radius of `pixel_size / 2`.
    """
    # Get the UV coordinates of the pixels that have passed the threshold.
    uv_pixels = get_threshold_pixels_uv(image_path, threshold)

    # Get the width(U = X) and height(V = -Y) of the image.
    w_pixels, h_pixels = Image.open(image_path).size
    w_meters = float(w_pixels) * pixel_size
    h_meters = float(h_pixels) * pixel_size

    # Convert the UV coordinates to XY coordinates.
    # The current XY origin is at the UV origin.
    xy_pixels = uv_pixels * pixel_size
    xy_pixels[:, 1] = -xy_pixels[:, 1]

    # Origin of the image in XY coordinates.
    origin_in_xy = (origin_in_uv[0] * w_meters, -origin_in_uv[1] * h_meters)

    # Set the XY origin to the desired location.
    xy_pixels -= origin_in_xy

    # Convert the XY coordinates to obstacles.
    radius = pixel_size / 2.0
    obstacles = [collision.Circle(collision.Vector(xy[0], xy[1]), radius) for xy in xy_pixels]
    return obstacles


def convert_bounds_to_obstacles(bounds: Tuple[float, float, float, float]) -> List[collision.Poly]:
    """Convert the bounds to obstacles.

    Args:
        bounds (tuple[float, float, float, float]): Bounds in the format (xmin, xmax, ymin, ymax).

    Returns:
        list: List of obstacles.
    """
    xmin, xmax, ymin, ymax = bounds
    # Midpoint of the bounds
    xmid = (xmax + xmin) / 2.0
    ymid = (ymax + ymin) / 2.0
    # `t` is the thickness to avoid ZeroDivisionError
    # If the thickness is too small, you may not be able to see from the matplot.
    # Increasing the thickness makes it thicker on the outside.
    t = 0.05
    half_t = t / 2.0
    # `ex` is the extended length to fill corners
    ex = 2.0 * t
    # Boundary obstacles: left, right, bottom, top
    L = collision.Poly.from_box(collision.Vector(xmin - half_t, ymid), t, ymax - ymin + ex)
    R = collision.Poly.from_box(collision.Vector(xmax + half_t, ymid), t, ymax - ymin + ex)
    B = collision.Poly.from_box(collision.Vector(xmid, ymin - half_t), xmax - xmin, t)
    T = collision.Poly.from_box(collision.Vector(xmid, ymax + half_t), xmax - xmin, t)
    return [L, R, B, T]


def convert_line_to_points(p1: np.ndarray, p2: np.ndarray, min_resolution: float) -> np.ndarray:
    """Extract a line as a list of points.

    Args:
        p1 (np.ndarray): First point of the line. Shape: (2,).
        p2 (np.ndarray): Second point of the line. Shape: (2,).
        min_resolution (float): Minimum resolution of the line. The distance between two adjacent points <= min_resolution.

    Returns:
        np.ndarray: Array of points that represent the line. Shape: (num_points, 2).
    """
    # Calculate the distance between the points
    distance = np.linalg.norm(p2 - p1)

    # Calculate the number of points to extract
    num_points = np.ceil(distance / min_resolution).astype(int)

    # Calculate the points
    points = np.linspace(p1, p2, num_points, endpoint=False)
    return points


def extract_outline_as_points(src: Collidable, min_resolution: float) -> np.ndarray:
    """Extract the outline of the object as a list of points.

    Args:
        src (object): Object to extract the outline from.
        min_resolution (float): Minimum resolution of the outline. The distance between two adjacent points <= min_resolution.

    Returns:
        np.ndarray: Array of points that represent the outline of the object. Shape: (num_points, 2).
    """
    validate_collision_type(src)

    if isinstance(src, collision.Circle):
        return _extract_circle_outline_as_points(src, min_resolution)
    elif isinstance(src, (collision.Poly, collision.Concave_Poly)):
        return _extract_poly_outline_as_points(src, min_resolution)


def _extract_circle_outline_as_points(src: collision.Circle, min_resolution: float) -> np.ndarray:
    radius = src.radius
    num_points = np.ceil(2 * np.pi * radius / min_resolution).astype(int)
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = np.array([radius * np.cos(angles) + src.pos.x, radius * np.sin(angles) + src.pos.y]).T
    return points


def _extract_poly_outline_as_points(src: Union[collision.Poly, collision.Concave_Poly], min_resolution: float) -> np.ndarray:
    vertices: List[collision.Vector] = src.points
    vertices: np.ndarray = np.array([[v.x, v.y] for v in vertices])
    lines = [(v1, v2) for v1, v2 in zip(vertices, np.roll(vertices, -1, axis=0))]
    points = np.concatenate([convert_line_to_points(v1, v2, min_resolution) for v1, v2 in lines])
    return points


def validate_collision_type(instance: Collidable) -> None:
    """Validate that the instance is of a collidable type, otherwise raise an error.

    Note:
        Valid types are: collision.Circle, collision.Poly, collision.Concave_Poly

    Args:
        instance (object): Object to check.

    Raises:
        TypeError: If the object is not collidable.
    """
    valid_types = (collision.Circle, collision.Poly, collision.Concave_Poly)
    if not isinstance(instance, valid_types):
        raise TypeError(f"Invalid obstacle type: {type(instance)}\nObstacle must be one of {valid_types}.")


if __name__ == "__main__":
    # Example usage:
    image_path = "../../examples/Berlin_0_512.png"
    w_pixels, h_pixels = Image.open(image_path).size
    print(f"width: {w_pixels}, height: {h_pixels}")

    threshold = 128
    uv_pixels = get_threshold_pixels_uv(image_path, threshold)
    print(uv_pixels)
