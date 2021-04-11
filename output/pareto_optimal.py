#!/usr/bin/env python
"""
Program to search for pareto-optimal MLPs from rmse_time.csv
"""

# import standard modules
import os
from csv import reader
import numpy as np
from scipy.spatial import ConvexHull

if __name__ == "__main__" :
    mlp_dict = {}
    points = []
    csv_path = os.getenv("HOME")+"/mlp-Fe/output/Fe/rmse_time.csv"
    f = open(csv_path)
    csv_iter = reader(f)
    for i, point in enumerate(csv_iter):
        mlp_dict[i] = point[0]
        points.append([float(point[1]), float(point[2])])
    f.close()

    pareto_list = []
    points = np.array(points)
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        x1 = points[simplex[0]][0]
        x2 = points[simplex[1]][0]
        y1 = points[simplex[0]][1]
        y2 = points[simplex[1]][1]
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0 :
            pareto_list.append(simplex)

    for line in pareto_list:
        print(mlp_dict[line[0]]+", "+mlp_dict[line[1]])
