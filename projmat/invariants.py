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

# import modules to use mathematical functions
import sympy
from sympy.physics.wigner import clebsch_gordan as cg

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
                c1 = cg(orbit_ns[0], orbit_ns[1], orbit_ns[2], i[0], i[1], -i[2])
                c2 = cg(orbit_ns[0], orbit_ns[1], orbit_ns[2], j[0], j[1], -j[2])
                df.loc[i, j] = sympy.N((-1)**(i[2]-j[2]) * 1/(2 * orbit_ns[2] + 1) * c1 * c2)
                df = np.array(df).astype(np.float64)
                df = pd.DataFrame(df, index=mindex, columns=mindex)
    if len(orbit_ns) == 4:
        for i in mindex:
            for j in mindex:
                p = 0
                for l in range(abs(orbit_ns[0]-orbit_ns[1]), orbit_ns[0]+orbit_ns[1]):
                    c1 = cg(orbit_ns[0], orbit_ns[1], l, i[0], i[1], i[0]+i[1])
                    c2 = cg(orbit_ns[0], orbit_ns[1], l, j[0], j[1], j[0]+j[1])
                    c3 = cg(orbit_ns[2], l, orbit_ns[3], i[2], i[0]+i[1], -i[3])
                    c4 = cg(orbit_ns[2], l, orbit_ns[3], j[2], j[0]+j[1], -j[3])
                    p += (-1)**(i[3]-j[3])*1/(2*orbit_ns[3]+1)*c1*c2*c3*c4
                df = np.array(df).astype(np.float64)
                df = pd.DataFrame(df, index=mindex, columns=mindex)
    if len(orbit_ns) == 5:
        for i in mindex:
            for j in mindex:
                p = 0
                for l in range(abs(orbit_ns[0]-orbit_ns[1]), orbit_ns[0]+orbit_ns[1]):
                    for L in range(abs(orbit_ns[2]-l), orbit_ns[2]+l):
                        c1 = cg(orbit_ns[0], orbit_ns[1], l, i[0], i[1], i[0]+i[1])
                        c2 = cg(orbit_ns[0], orbit_ns[1], l, j[0], j[1], j[0]+j[1])
                        c3 = cg(orbit_ns[2], l, L, i[2], i[0]+i[1], i[0]+i[1]+i[2])
                        c4 = cg(orbit_ns[2], l, L, j[2], j[0]+j[1], j[0]+j[1]+j[2])
                        c5 = cg(orbit_ns[3], L, orbit_ns[4], i[3], i[0]+i[1]+i[2], -i[4])
                        c6 = cg(orbit_ns[3], L, orbit_ns[4], j[3], j[0]+j[1]+j[2], -j[4])
                        p += (-1)**(i[4]-j[4])*1/(2*orbit_ns[4]+1)*c1*c2*c3*c4*c5*c6
                df = np.array(df).astype(np.float64)
                df = pd.DataFrame(df, index=mindex, columns=mindex)
    if len(orbit_ns) == 6:
        for i in mindex:
            for j in mindex:
                p = 0
                for l in range(abs(orbit_ns[0]-orbit_ns[1]), orbit_ns[0]+orbit_ns[1]):
                    for L in range(abs(orbit_ns[2]-l), orbit_ns[2]+l):
                        for S in range(abs(orbit_ns[3]-L), orbit_ns[3]+L):
                            c1 = cg(orbit_ns[0], orbit_ns[1], l, i[0], i[1], i[0]+i[1])
                            c2 = cg(orbit_ns[0], orbit_ns[1], l, j[0], j[1], j[0]+j[1])
                            c3 = cg(orbit_ns[2], l, L, i[2], i[0]+i[1], i[0]+i[1]+i[2])
                            c4 = cg(orbit_ns[2], l, L, j[2], j[0]+j[1], j[0]+j[1]+j[2])
                            c5 = cg(orbit_ns[3], L, S, i[3], i[0]+i[1]+i[2], i[0]+i[1]+i[2]+i[3])
                            c6 = cg(orbit_ns[3], L, S, j[3], j[0]+j[1]+j[2], j[0]+j[1]+j[2]+j[3])
                            c7 = cg(orbit_ns[4], S, orbit_ns[5], i[4], i[0]+i[1]+i[2]+i[3], -i[5])
                            c8 = cg(orbit_ns[4], S, orbit_ns[5], j[4], j[0]+j[1]+j[2]+j[3], -j[5])
                            p += (-1)**(i[5]-j[5])*1/(2*orbit_ns[5]+1)*c1*c2*c3*c4*c5*c6*c7*c8
                df = np.array(df).astype(np.float64)
                df = pd.DataFrame(df, index=mindex, columns=mindex)
    # calculate eigenvalue and eigenvector
    eig = np.linalg.eig(df)
    evecs = eig[1][:, np.isclose(eig[0], 1)]
    if np.any(evecs):
        ser = pd.Series(evecs.reshape(-1), index=mindex)
        print(ser)
