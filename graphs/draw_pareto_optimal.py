#!/usr/bin/env python
"""
Program to draw the lines connecting pareto optimal MLPs and points of rmse_time.csv
"""

# import standard modules
import os
from csv import reader
import matplotlib.pyplot as plt

if __name__ == "__main__" :
    dir_path = os.getenv("HOME")+"/mlp-Fe/output/Fe/"
    points_path = dir_path+"rmse_time.csv"
    lines_path = dir_path+"pareto_optimal.csv"
    f = open(points_path)
    mlp_dict = {point[0]: [float(point[1]), float(point[2])] for point in reader(f)}
    f.close()
    points = list(mlp_dict.values())
    plt.scatter(points[:, 0], points[:, 1])
