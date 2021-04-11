#!/usr/bin/env python
"""
Program to draw the lines connecting pareto optimal MLPs and points of rmse_time.csv
"""

# import standard modules
import os
from csv import reader
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__" :
    dir_path = os.getenv("HOME")+"/mlp-Fe/output/Fe/"
    points_path = dir_path+"rmse_time.csv"
    lines_path = dir_path+"pareto_optimal.csv"
    f = open(points_path)
    mlp_dict = {point[0]: [float(point[1]), float(point[2])] for point in reader(f)}
    f.close()
    points = np.array(list(mlp_dict.values()))

    fig, ax = plt.subplots()
    f = open(lines_path)
    for line in reader(f) :
        x = [mlp_dict[line[0]][0], mlp_dict[line[1]][0]]
        y = [mlp_dict[line[0]][1], mlp_dict[line[1]][1]]
        ax.plot(x, y, 'r-')
    f.close()
    ax.scatter(points[:, 0], points[:, 1], s=2, color='k', marker=".")
    plt.show()
