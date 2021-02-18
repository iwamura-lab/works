#!/usr/bin/env python
"""
Program to draw and save the graph of Lennard Jones potential
"""

# import module to handle array
import numpy as np

# import module to draw graph
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ax = plt.axes()
    r = np.linspace(0.1, 2, 950)
    cut_off = 0.5 * (np.cos(np.pi * r/4.5) + 1)
    ax.plot(r, 1/r**12 - 2/r**6)
    ax.set(xlim=(0.6, 2.0), ylim=(-1.5, 2.0))
    plt.savefig("results/graphs/lennardJones.png")
