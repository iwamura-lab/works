#!/usr/bin/env python
"""
Program to search for pareto-optimal MLPs from rmse_time.csv
"""

# import standard modules
import os
import argparse
from csv import reader
import numpy as np
from scipy.spatial import ConvexHull

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str,
                       default="rmse_time.csv", help="input file name")
    args = parser.parse_args()

    mlp_dict = {}
    points = []
    csv_path = os.getenv("HOME")+"/mlp-Fe/output/Fe/"+args.file
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

        # select negative slope lines below all the points
        if slope < 0 :
            bool_list = [(point[1] - y1 - slope * (point[0] - x1)) >= -0.1 for point in points]
            if np.all(bool_list) :
                pareto_list.append(simplex)

    for line in pareto_list:
        print(mlp_dict[line[0]]+","+mlp_dict[line[1]])
