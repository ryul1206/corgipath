import matplotlib.pyplot as plt
from matplotlib.colors import Colormap


def pick_color(x, theme="rainbow"):
    """Pick a color from a theme.

    Note:
        The theme should be a matplotlib colormap. See:
            * [Matplotlib colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)
            * [Matplotlib colors](https://matplotlib.org/stable/tutorials/colors/colors.html)
    """
    if x < 0.0 or x > 1.0:
        raise ValueError("x should be in [0, 1]")
    cmap: Colormap = plt.get_cmap(theme)
    return cmap(x)  # RGBA


def auto_scale(ax: plt.Axes):
    """Auto scale the plot.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw.
    """
    ax.relim(visible_only=True)  # Recompute the ax.dataLim
    ax.set_aspect("equal", adjustable="box")  # Set aspect ratio to 1.0
    ax.autoscale_view()  # Update ax.viewLim using the new dataLim
