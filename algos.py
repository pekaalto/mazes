import heapq
from collections import deque
import numpy as np


def bfs(maze, start, goal):
    return _bfs_or_dfs(
        maze,
        start,
        goal,
        get_queue=lambda start: deque([start]),
        add_nodes_to_queue=lambda queue, nodes: queue.extendleft(nodes),
    )


def dfs(maze, start, goal):
    return _bfs_or_dfs(
        maze,
        start,
        goal,
        get_queue=lambda start: [start],
        add_nodes_to_queue=lambda queue, nodes: queue.extend(nodes),
    )


def _bfs_or_dfs(maze, start, goal, get_queue, add_nodes_to_queue):
    explored = []
    parents = {start: None}
    queue = get_queue(start)

    finished = False
    while queue:
        v = queue.pop()

        explored.append(v)
        if v == goal:
            finished = True
            break

        new_neighbors = get_neighbors(v, maze).difference(parents)
        for n in new_neighbors:
            parents[n] = v

        add_nodes_to_queue(queue, shuffle(new_neighbors))

    path = None if not finished else create_path(parents, start, goal)
    assert len(explored) == len(set(explored))

    return finished, explored, path


def astar(maze, start, goal, heuristic):
    explored = []
    parents = {}
    heap = [(0, start)]
    dist_from_beginning = {start: 0}

    finished = False
    while heap:
        v = heapq.heappop(heap)[1]

        explored.append(v)
        if v == goal:
            finished = True
            break

        neighbors = get_neighbors(v, maze)
        for node in shuffle(neighbors):
            dist_new = dist_from_beginning[v] + 1
            if dist_new < dist_from_beginning.get(node, float("inf")):
                dist_from_beginning[node] = dist_new
                heapq.heappush(heap, (dist_new + heuristic(node), node))
                parents[node] = v

    path = None if not finished else create_path(parents, start, goal)

    return finished, explored, path


def get_neighbors(cell, maze):
    neighbors = set()
    for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ii = cell[0] + i
        jj = cell[1] + j
        if ii < 0 or jj < 0:
            continue

        try:
            item = maze[ii][jj]
            if item:
                neighbors.add((ii, jj))
        except IndexError:
            pass

    return neighbors


def shuffle(nodes):
    nodes = list(nodes)
    np.random.shuffle(nodes)
    return nodes


def create_path(parents, start, goal):
    path = []
    node = goal
    while True:
        path.append(node)
        node = parents[node]
        if node == start:
            break

    return list(reversed(path))


def manhattan_heuristic(size):
    def f(x):
        return (size - 1 - x[0]) + (size - 1 - x[1])

    return f
