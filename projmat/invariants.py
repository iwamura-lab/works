'''
Program to compute rotational invariants of SO3 group by using
projection operator method
This deals with only spherical harmonics, but may be improved
in order to use other functions as basis functions in the near future
'''

# set python interpreter(2 or 3 ?)
# !/usr/bin/env/python
# -*- coding: UTF-8 -*-

# import standard modules
from itertools import product

# import modules to operate matrix
import numpy as np

# import modules to handle enormous datasets
import pandas as pd

# import modules to use mathematical functions
import sympy
from sympy.physics.wigner import clebsch_gordan as cg

def make_index(orbit_ls):
    """
    Docstring:
    make_index(orbit_ns)

    Make multi-index from the input orbit_ns
    """
    mindex = []
    for l in orbit_ls:
        mindex.append([m for m in range(-l, l+1)])
    return mindex

if __name__ == "__main__":
    # get the l list
    print("Enter the azimuthal quantum number list of seed functions")
    lis = input("such as l1, l2 ,... , lp:")
    lis = lis.split(",")
    lis = [int(l) for l in lis]
    # calculate total sum of m
    lsum = 1
    for i in lis:
        lsum *= 2 * i + 1
    if len(lis) == 2 and lis[0] == lis[1]:
        id_list = list(product(range(-lis[0], lis[0]+1), range(-lis[1], lis[1]+1)))
        pmat = np.zeros((lsum, lsum))
        mid = {c: i for i, c in enumerate(id_list)}
        for cm in id_list:
            for rm in id_list:
                if cm[0] == -cm[1] and rm[0] == -rm[1]:
                    pmat[mid[cm], mid[rm]] = (-1)**(cm[1]-rm[1])*1/(2*lis[0]+1)

    if len(lis) == 3:
        for i in mindex:
            for j in mindex:
                c1 = cg(orbit_ns[0], orbit_ns[1], orbit_ns[2], i[0], i[1], -i[2])
                c2 = cg(orbit_ns[0], orbit_ns[1], orbit_ns[2], j[0], j[1], -j[2])
                df.loc[i, j] = sympy.N((-1)**(i[2]-j[2]) * 1/(2 * orbit_ns[2] + 1) * c1 * c2)
        df = np.array(df).astype(np.float64)
        df = pd.DataFrame(df, index=mindex, columns=mindex)
    if len(lis) == 4:
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
    if len(lis) == 5:
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
    if len(lis) == 6:
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
    # calculate eigenvalue and eigenvector
    # import pdb
    # pdb.set_trace()
    eig = np.linalg.eig(pmat)
    evecs = eig[1][:, np.isclose(eig[0], 1)]
    if np.any(evecs):
        df_evec = pd.DataFrame(evecs, index=mindex)
        print(df_evec)
