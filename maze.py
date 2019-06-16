import numpy as np


def create_maze(size, n_walls, length_walls=0.3):
    maze = np.ones((size, size), dtype=bool)

    for i in range(n_walls):
        row, col = np.random.randint(0, size - 1, 2)

        wall = np.random.poisson(length_walls) * np.random.choice([-1, 1])

        if np.random.choice([True, False]):
            row = make_range(row, wall, size)
        else:
            col = make_range(col, wall, size)

        maze[row, col] = False

    maze[0, 0] = True
    maze[size - 1, size - 1] = True
    return maze


def make_range(start, length, m):
    d = np.clip(np.array([start + length, start]), 0, m - 1)
    return list(range(d.min(), d.max() + 1))
