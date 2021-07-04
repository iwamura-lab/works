#!/usr/bin/env python
"""
Program to calculate RMSE in cohesive energy of test data
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, \
        default="MLP1", help="Potential data directory for pareto MLP")
    parser.add_argument("-p", "--plot", action="store_true", \
                        help="Plot diagonal line.")
    args = parser.parse_args()

    file_path = os.getenv("HOME")+"/mlp-Fe/pareto/"+\
                args.file+"/predictions/energy.out.test_vasprun_Fe"
    with open(file_path) as f:
        data = f.readlines()
    dif_list = [float(line.split()[2])**2 for line in data[1:]]
    rmse = (sum(dif_list)/len(dif_list))**(1/2)
    print("{:.6f} ev/atom".format(rmse))
    if args.plot :
        points = np.array([list(map(float, line.split()[:2])) for line in data[1:]])
        points = points * (-1)
        ax = plt.axes()
        ax.grid()
        ax.set(xlim=(0, 6), ylim=(0, 6))
        x = np.linspace(0, 6, 50)
        ax.plot(x, x, c="black", ls=":")
        plt.scatter(points[:, 0], points[:, 1], s=4**2, c="black")
        ticks = [i for i in range(7)]
        plt.xticks(ticks, fontsize=20, position=(0.0, -0.02))
        plt.yticks(ticks, fontsize=20, x=-0.02)
        plt.show()
