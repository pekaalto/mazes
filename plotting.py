import matplotlib.pyplot as plt
import numpy as np
import math


class Plotter:
    def __init__(
        self,
        maze,
        explored,
        path,
        end_frames = 0,
        frame_skip_path=2,
        frame_skip_search=4,
        explored_color=2,
        path_color=3,
        color_limits=(0, 3.5),
        color_map="jet",
        wall_color=0,
        way_color=1,
    ):

        self.explored_i, self.explored_j = _unzip(explored)
        self.path_i, self.path_j = _unzip(path)
        self.maze = _get_int_maze(maze, wall_color, way_color)
        self.frame_skip_path = frame_skip_path
        self.frame_skip_search = frame_skip_search
        self.explored_color = explored_color
        self.path_color = path_color
        self.color_limits = color_limits
        self.color_map = color_map
        self.end_frames = end_frames
        self.im = None

    @property
    def total_frames(self):
        return self._total_search_frames + self._total_path_frames + 1

    @property
    def _total_search_frames(self):
        return math.ceil(len(self.explored_i) / self.frame_skip_search) + 1

    @property
    def _total_path_frames(self):
        return math.ceil(len(self.path_i) / self.frame_skip_path) + 1 + self.end_frames

    def init_fig(self, title=None):
        self.fig, self.ax = plt.subplots(figsize=(4,4))
        if title is not None:
            self.ax.set_title(title)

        self.init_fn()
        self.fig.tight_layout()

    def init_fn(self):
        im = self.ax.imshow(
            self.maze,
            cmap=self.color_map,
            vmin=self.color_limits[0],
            vmax=self.color_limits[1],
            interpolation="none",
        )
        self.im = im

    def anim_fn(self, i):
        a = self.im.get_array()

        if i < self._total_search_frames:
            f = i * self.frame_skip_search
            a[self.explored_i[:f], self.explored_j[:f]] = self.explored_color
        else:
            f = (i - self._total_search_frames) * self.frame_skip_path
            a[self.path_i[:f], self.path_j[:f]] = self.path_color

        self.im.set_array(a)
        return [self.im]


def _get_int_maze(maze, wall_color, way_color):
    assert maze.dtype == np.dtype("bool")
    new_maze = np.zeros_like(maze, dtype=np.int32)
    new_maze[np.where(maze)] = way_color
    new_maze[np.where(~maze)] = wall_color
    return new_maze


def _unzip(x):
    return tuple(np.array(k) for k in (zip(*x)))
