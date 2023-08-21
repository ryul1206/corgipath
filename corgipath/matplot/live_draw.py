from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class LiveDrawOption:
    """Option for LiveDraw

    Attributes:
        draw_func (callable): Function to draw the option. If `None`, it will not draw anything. Default is None.
            Do not call this function directly. Use the `draw` method instead.
        pause_before (float): Pause before drawing. Default is 0.0.
        pause_after (float): Pause after drawing. Default is 0.0.
        wait_key (bool): Wait for key press. Default is False. If True, it will execute `plt.waitforbuttonpress()`.
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.waitforbuttonpress.html for details.
    """
    draw_func: callable = None
    pause_before: float = 0.0
    pause_after: float = 0.0
    wait_key: bool = False

    def draw(self, *args, **kwargs):
        """Draw the option

        This function will call the draw_func with the given arguments.
        """
        plt.pause(0.0001)  # Without this line, the figure will not be updated.
        if self.pause_before > 0.0:
            plt.pause(self.pause_before)
        self.draw_func(*args, **kwargs)
        if self.pause_after > 0.0:
            plt.pause(self.pause_after)
        if self.wait_key:
            plt.waitforbuttonpress()
