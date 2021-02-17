#!/usr/bin/env python
"""
Program to draw and save the graph of 2D normal distribution
"""

# import module to handle array
import numpy as np

# import module to draw graph
import matplotlib.pyplot as plt

def f(x, y, sigma):
    """Return 2d gaussian function

    Args:
        x (ndarray): x coordinate
        y (ndarray): y coordinate
        sigma (float): standard deviation

    Returns:
        ndarray: function value corresponding to z
    """
    coef = 1/(2*np.pi*sigma**2)
    return coef * np.exp(-x**2/(2*sigma**2)) * np.exp(-y**2/(2*sigma**2))

if __name__ == "__main__":
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x = y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y, 1.0)
    ax.set(xlim=(-5, 5), ylim=(-5, 5), xlabel="x", ylabel="y")
    ax.plot_wireframe(X, Y, Z)
    plt.savefig("results/graphs/2d_gauss.png")
