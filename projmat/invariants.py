'''
Program to compute rotational invariants of SO3 group by using
projection operator method
This deals with only spherical harmonics, but may be improved
in order to use other functions as basis functions in the near future
'''

# set python interpreter(2 or 3 ?)
# !/usr/bin/env/python
# -*- coding: UTF-8 -*-

# import modules to operate matrix
import numpy as np

# import modules to handle enormous datasets
import pandas as pd

# import modules to do mathematical operations
import sympy
from sympy.physics.wigner import clebsch_gordan

def make_index(orbit_ns):
    """
    Docstring:
    make_index(orbit_ns)

    Make multi-index from the input orbit_ns
    """
    mindex = []
    for l in orbit_ns:
        mindex.append([m for m in range(-l, l+1)])
    return mindex

if __name__ == "__main__":
    # get the l list
    print("Enter the azimuthal quantum number list of seed functions")
    orbit_ns = input("such as l1, l2 ,... , lp:")
    orbit_ns = orbit_ns.split(",")
    orbit_ns = [int(l) for l in orbit_ns]
    # calculate total sum of m
    lsum = 1
    for i in orbit_ns:
        lsum *= 2 * i + 1
    # make multi-index for DataFrame and make DataFrame
    mlis = make_index(orbit_ns)
    mindex = pd.MultiIndex.from_product(mlis)
    df = pd.DataFrame(np.zeros((lsum, lsum)), index=mindex, columns=mindex)
    if len(orbit_ns) == 2 and orbit_ns[0] == orbit_ns[1]:
        for i in mindex:
            for j in mindex:
                if i[0] == -i[1] and j[0] == -j[1]:
                    df.loc[i, j] = (-1)**(i[1]-j[1]) * 1/(2 * orbit_ns[0] + 1)
    if len(orbit_ns) == 3:
        for i in mindex:
            for j in mindex:
                c1 = clebsch_gordan(orbit_ns[0], orbit_ns[1], orbit_ns[2], i[0], i[1], -i[2])
                c2 = clebsch_gordan(orbit_ns[0], orbit_ns[1], orbit_ns[2], j[0], j[1], -j[2])
                df.loc[i, j] = sympy.N((-1)**(i[2]-j[2]) * 1/(2 * orbit_ns[2] + 1) * c1 * c2)
    # calculate eigenvalue and eigenvector
    print(df.loc[(1, 1, -2), (1, 1, -2)])
    eig = np.linalg.eig(df)
    evecs = eig[1][:, np.isclose(eig[0], 1)]
    ser = pd.Series(evecs.reshape(-1), index=mindex)
    print(ser)
