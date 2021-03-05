#!/usr/bin/env python
"""
Program to draw and save the graph of radial functions
"""

# import module to handle array
import numpy as np

# import module to draw graph
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fig, ax = plt.subplots()
    t = np.linspace(0, 4.5, 1000)
    cut_off = 0.5 * (np.cos(np.pi * t/4.5) + 1)
    y1 = np.exp(-(t - 1)**2) * cut_off
    y2 = np.exp(-(t - 2)**2) * cut_off
    y3 = np.exp(-(t - 3)**2) * cut_off
    y4 = np.exp(-(t - 4)**2) * cut_off
    c1, c2, c3, c4 = "blue", "green", "red", "m"
    ax.set(xlim=(0, 5), ylim=(0, 1.0))
    ax.plot(t, y1, color=c1)
    ax.plot(t, y2, color=c2)
    ax.plot(t, y3, color=c3)
    ax.plot(t, y4, color=c4)
    plt.savefig("results/graphs/gcomp.png")
