import numpy as np
from ._base import BasePlanner


class Astar(BasePlanner):
    def __init__(self):
        super().__init__()

    def solve(self, start, goal, user_cfg) -> list:
        raise NotImplementedError
