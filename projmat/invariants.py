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
import sys
from math import factorial, sqrt

# import modules to operate matrix
import numpy as np

# import modules to use mathematical functions
# from sympy.physics.wigner import clebsch_gordan as my_clebsch

def my_clebsch(j1, j2, j3, m1, m2, m3):
    '''
    Docstring:
    my_clebsch(j1, j2, j3, m1, m2, m3)

    Calculate clebsch_gordan coefficient from input arguments.
    Sympy is the module to use symbolic algebra, so its behavior is too slow
    when using high order matrix.
    When only the result value is needed, this function is useful.
    '''
    if m1 + m2 != m3:
        return 0
    max1 = j2 + j3 + m1
    max2 = j3 - j1 + j2
    max3 = j3 + m3
    min1 = m1 - j1
    min2 = j2 - j1 + m3
    vmax = min(max1, max2, max3)
    vmin = max(min1, min2, 0)
    deno = [j3+j1-j2, j3-j1+j2, j1+j2-j3, j3+m3, j3-m3]
    nume = [j1+j2+j3+1, j1-m1, j1+m1, j2-m2, j2+m2]
    for i in deno:
        if i < 0:
            return 0
    for i in nume:
        if i < 0:
            return 0
    dfac = 1
    for i in deno:
        dfac *= factorial(i)
    nfac = 1
    for i in nume:
        nfac *= factorial(i)
    cf = sqrt((2*j3+1)*dfac / nfac)
    csum = 0
    for v in range(vmin, vmax+1):
        csum += ((-1)**(v+j2+m2)*factorial(j2+j3+m1-v)*factorial(j1-m1+v)/
                 (factorial(v)*factorial(j3-j1+j2-v)*factorial(j3+m3-v)*factorial(v+j1-j2-m3)))
    return cf * csum

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
    pmat = np.zeros((lsum, lsum))
    # division to cases
    if len(lis) == 2 and lis[0] == lis[1]:
        id_list = list(product(range(-lis[0], lis[0]+1), range(-lis[1], lis[1]+1)))
        mid = {c: i for i, c in enumerate(id_list)}
        for cm in id_list:
            for rm in id_list:
                if cm[0] == -cm[1] and rm[0] == -rm[1]:
                    pmat[mid[cm], mid[rm]] = (-1)**(cm[1]-rm[1])/(2*lis[0]+1)
    elif len(lis) == 2 and lis[0] != lis[1]:
        print("Projection matrix equals to zero matrix")
        sys.exit(0)
    if len(lis) == 3:
        id_list = list(product(range(-lis[0], lis[0]+1), range(-lis[1], lis[1]+1),
                               range(-lis[2], lis[2]+1)))
        mid = {c: i for i, c in enumerate(id_list)}
        for cm in id_list:
            for rm in id_list:
                c1 = my_clebsch(lis[0], lis[1], lis[2], cm[0], cm[1], -cm[2])
                c2 = my_clebsch(lis[0], lis[1], lis[2], rm[0], rm[1], -rm[2])
                pmat[mid[cm], mid[rm]] = (-1)**(cm[2]-rm[2])/(2*lis[2]+1)*c1*c2
        pmat = pmat.astype(np.float64)
    if len(lis) == 4:
        id_list = list(product(range(-lis[0], lis[0]+1), range(-lis[1], lis[1]+1),
                               range(-lis[2], lis[2]+1), range(-lis[3], lis[3]+1)))
        mid = {c: i for i, c in enumerate(id_list)}
        for cm in id_list:
            for rm in id_list:
                p = 0
                for l in range(abs(lis[0]-lis[1]), lis[0]+lis[1]):
                    c1 = my_clebsch(lis[0], lis[1], l, cm[0], cm[1], cm[0]+cm[1])
                    c2 = my_clebsch(lis[0], lis[1], l, rm[0], rm[1], rm[0]+rm[1])
                    c3 = my_clebsch(lis[2], l, lis[3], cm[2], cm[0]+cm[1], -cm[3])
                    c4 = my_clebsch(lis[2], l, lis[3], rm[2], rm[0]+rm[1], -rm[3])
                    p += (-1)**(rm[3]-cm[3])/(2*lis[3]+1)*c1*c2*c3*c4
                pmat[mid[cm], mid[rm]] = p
        pmat = pmat.astype(np.float64)
    if len(lis) == 5:
        id_list = list(product(range(-lis[0], lis[0]+1), range(-lis[1], lis[1]+1),
                               range(-lis[2], lis[2]+1), range(-lis[3], lis[3]+1),
                               range(-lis[4], lis[4]+1)))
        mid = {c: i for i, c in enumerate(id_list)}
        for cm in id_list:
            for rm in id_list:
                p = 0
                for l in range(abs(lis[0]-lis[1]), lis[0]+lis[1]):
                    for L in range(abs(lis[2]-l), lis[2]+l):
                        c1 = my_clebsch(lis[0], lis[1], l, cm[0], cm[1], cm[0]+cm[1])
                        c2 = my_clebsch(lis[0], lis[1], l, rm[0], rm[1], rm[0]+rm[1])
                        c3 = my_clebsch(lis[2], l, L, cm[2], cm[0]+cm[1], cm[0]+cm[1]+cm[2])
                        c4 = my_clebsch(lis[2], l, L, rm[2], rm[0]+rm[1], rm[0]+rm[1]+rm[2])
                        c5 = my_clebsch(lis[3], L, lis[4], cm[3], cm[0]+cm[1]+cm[2], -cm[4])
                        c6 = my_clebsch(lis[3], L, lis[4], rm[3], rm[0]+rm[1]+rm[2], -rm[4])
                        p += (-1)**(cm[4]-rm[4])*1/(2*lis[4]+1)*c1*c2*c3*c4*c5*c6
                pmat[mid[cm], mid[rm]] = p
        pmat = pmat.astype(np.float64)
    if len(lis) == 6:
        id_list = list(product(range(-lis[0], lis[0]+1), range(-lis[1], lis[1]+1),
                               range(-lis[2], lis[2]+1), range(-lis[3], lis[3]+1),
                               range(-lis[4], lis[4]+1), range(-lis[5], lis[5]+1)))
        mid = {c: i for i, c in enumerate(id_list)}
        for cm in id_list:
            for rm in id_list:
                p = 0
                for l in range(abs(lis[0]-lis[1]), lis[0]+lis[1]):
                    for L in range(abs(lis[2]-l), lis[2]+l):
                        for S in range(abs(lis[3]-L), lis[3]+L):
                            c1 = my_clebsch(lis[0], lis[1], l, cm[0], cm[1], cm[0]+cm[1])
                            c2 = my_clebsch(lis[0], lis[1], l, rm[0], rm[1], rm[0]+rm[1])
                            c3 = my_clebsch(lis[2], l, L, cm[2], cm[0]+cm[1], cm[0]+cm[1]+cm[2])
                            c4 = my_clebsch(lis[2], l, L, rm[2], rm[0]+rm[1], rm[0]+rm[1]+rm[2])
                            c5 = my_clebsch(lis[3], L, S, cm[3], cm[0]+cm[1]+cm[2],
                                            cm[0]+cm[1]+cm[2]+cm[3])
                            c6 = my_clebsch(lis[3], L, S, rm[3], rm[0]+rm[1]+rm[2],
                                            rm[0]+rm[1]+rm[2]+rm[3])
                            c7 = my_clebsch(lis[4], S, lis[5], cm[4],
                                            cm[0]+cm[1]+cm[2]+cm[3], -cm[5])
                            c8 = my_clebsch(lis[4], S, lis[5], rm[4],
                                            rm[0]+rm[1]+rm[2]+rm[3], -rm[5])
                            p += (-1)**(cm[5]-rm[5])*1/(2*lis[5]+1)*c1*c2*c3*c4*c5*c6*c7*c8
                pmat[mid[cm], mid[rm]] = p
        pmat = pmat.astype(np.float64)
    # calculate eigenvalue and eigenvector
    # insert debugger
    # import pdb
    # pdb.set_trace()
    eig = np.linalg.eig(pmat)
    evecs = eig[1][:, np.isclose(eig[0], 1)]
    if np.any(evecs):
        print(evecs)
