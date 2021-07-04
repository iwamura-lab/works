#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, \
        default="MLP1", help="Potential data directory for pareto MLP")
    args = parser.parse_args()

    file_path = os.getenv("HOME")+"/mlp-Fe/pareto/"+\
                args.file+"/predictions/energy.out.test_vasprun_Fe"
    with open(file_path) as f:
        data = f.readlines()
    points = np.array([list(map(float, line.split()[:2])) for line in data[1:]])
    points = points * (-1)
    ax = plt.axes()
    ax.grid()
    ax.set(xlim=(0, 5), ylim=(0, 5))
    x = np.linspace(0, 5, 50)
    ax.plot(x, x, c="black", ls=":")
    plt.scatter(points[:, 0], points[:, 1], s=4**2, c="black")
    ticks = [i for i in range(6)]
    plt.xticks(ticks, fontsize=20, position=(0.0, -0.02))
    plt.yticks(ticks, fontsize=20, x=-0.02)
    plt.show()
