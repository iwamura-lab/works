"""
Program to draw the graph of radial functions
"""

# set python interpreter(2 or 3 ?)
# !/usr/bin/python3
# -*- coding: UTF-8 -*-

# import module to handle array
import numpy as np

# import module to draw graph
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fig, ax = plt.subplots()
    t = np.linspace(0, 4.5, 1000)
    cut_off = 0.5 * (np.cos(np.pi * t/4.5) + 1)
    y1 = np.exp(-(t - 0)**2) * cut_off
    y2 = np.exp(-(t - 1)**2) * cut_off
    y3 = np.exp(-(t - 2)**2) * cut_off
    y4 = np.exp(-(t - 3)**2) * cut_off
    y5 = np.exp(-(t - 4)**2) * cut_off
    y6 = np.exp(-(t - 5)**2) * cut_off
    c1, c2, c3, c4, c5, c6 = "blue", "green", "red", "black", "pink", "purple"
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.set_title('gauss function')
    ax.grid()
    ax.plot(t, y1, color=c1)
    ax.plot(t, y2, color=c2)
    ax.plot(t, y3, color=c3)
    ax.plot(t, y4, color=c4)
    ax.plot(t, y5, color=c5)
    ax.plot(t, y6, color=c6)
    plt.show()
