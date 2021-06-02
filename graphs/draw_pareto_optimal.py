#!/usr/bin/env python
"""
Program to draw the lines connecting pareto optimal MLPs and points of rmse_time.csv
"""

# import standard modules
import os
import argparse
from csv import reader
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--rmse", "-r", type=str, default="rmse_time.csv",
                        help="data file of rmse and elapsed time")
    parser.add_argument("--pareto", "-p", type=str, default="pareto_optimal.csv",
                        help="data file about pareto optimal MLPs")
    args = parser.parse_args()

    dir_path = os.getenv("HOME")+"/mlp-Fe/output/Fe/"
    points_path = dir_path+args.rmse
    lines_path = dir_path+args.pareto
    f = open(points_path)
    mlp_dict = {point[0]: [float(point[1]), float(point[2])] for point in reader(f)}
    f.close()
    points = np.array(list(mlp_dict.values()))

    fig, ax = plt.subplots()
    f = open(lines_path)
    for line in reader(f) :
        x = [mlp_dict[line[0]][0], mlp_dict[line[1]][0]]
        y = [mlp_dict[line[0]][1], mlp_dict[line[1]][1]]
        ax.plot(x, y, 'r-', lw=3)
    f.close()
    ax.scatter(points[:, 0], points[:, 1], s=2, color='k', marker=".")
    ax.set(xlim=(0, 1.5), ylim=(6.5, 11.0), xticks=[0, 0.5, 1.0, 1.5], yticks=[6.5, 7.0, 8.0, 9.0, 10, 11])
    ax.tick_params(labelsize="30")
    plt.show()
