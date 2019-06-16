from collections import namedtuple
from functools import partial
from timeit import default_timer as timer
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from algos import bfs, dfs, astar, manhattan_heuristic
from maze import create_maze
from plotting import Plotter
import tabulate
import tqdm
import os

np.random.seed(0)

SIZE = 100
START = (0, 0)
GOAL = (SIZE - 1, SIZE - 1)
OBSTACLE_COUNT = SIZE ** 2 // 30
OBSTACLE_SIZE = 7

ALGOS = [
    ("DFS", partial(dfs, start=START, goal=GOAL)),
    ("BFS", partial(bfs, start=START, goal=GOAL)),
    (
        "A-star",
        partial(astar, start=START, goal=GOAL, heuristic=manhattan_heuristic(SIZE)),
    ),
]

Result = namedtuple(
    "Result", ["round", "finished", "n_explored", "length_path", "seconds", "algo"]
)


def simulate(n, max_good_mazes=float("inf")):
    results = []
    good_mazes = []
    for i in tqdm.tqdm(range(n), mininterval=1):
        maze = create_maze(SIZE, OBSTACLE_COUNT, OBSTACLE_SIZE)
        for algo_name, algo in ALGOS:
            time_start = timer()
            finished, explored_nodes, path = algo(maze)
            time_end = timer()
            r = Result(
                round=i,
                finished=finished,
                n_explored=len(explored_nodes),
                length_path=None if not finished else len(path),
                seconds=time_end - time_start,
                algo=algo_name,
            )
            results.append(r)
            if finished and algo_name == ALGOS[0][0]:
                good_mazes.append(maze)
                if len(good_mazes) >= max_good_mazes:
                    break

    results_df = pd.DataFrame(results)
    return results_df, good_mazes


def create_animation(title, algo, maze):
    finished, explored, path = algo(maze)
    assert finished, "goal not reachable"

    plotter = Plotter(
        maze, explored, path, frame_skip_path=8, frame_skip_search=32, end_frames=30
    )

    plotter.init_fig(title)

    anim = ani.FuncAnimation(
        plotter.fig,
        plotter.anim_fn,
        init_func=plotter.init_fn,
        frames=plotter.total_frames,
        interval=1,
        repeat=True,
    )
    return anim


result_df, good_mazes = simulate(1000)

result_agg = result_df.groupby(["finished", "algo"])[
    "seconds", "n_explored", "length_path"
].mean()

result_agg["count"] = result_df.groupby(["finished", "algo"]).size()
result_agg = result_agg.reset_index()

print(tabulate.tabulate(result_agg.values, result_agg.columns, tablefmt="pipe"))

output_path = "_files/anim3"
os.makedirs(output_path, exist_ok=True)

for algo_name, algo in ALGOS:
    anim = create_animation(algo_name, algo, good_mazes[0])
    anim.save(
        f"{output_path}/{algo_name.replace('*', '-star')}.gif", writer="imagemagick"
    )
    plt.close()
