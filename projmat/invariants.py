#!/usr/bin/env python
"""
Program to compute rotational invariants of SO3 group by using
projection operator method
This deals with only spherical harmonics, but may be improved
in order to use other functions as basis functions in the near future
"""

# import standard modules
import time
from itertools import product
import sys
from math import factorial, sqrt
import numpy as np

# import modules to use mathematical functions
# from sympy.physics.wigner import clebsch_gordan as clebsch_gordan

def clebsch_gordan(j1, j2, j3, m1, m2, m3):
    """Calculate clebsch gordan coefficient from the quantum numbers

    Args:
        j1 (int): angular momentum of 1st eigenstate before coupling
        j2 (int): angular momentum of 2nd eigenstate before coupling
        j3 (int): angular momentum after coupling
        m1 (int): projection angular momentum of 1st eigenstate before coupling
        m2 (int): projection angular momentum of 2nd eigenstate before coupling
        m3 (int): projection angular momentum after coupling

    Returns:
        float: clebsch gordan coefficient

    Sympy is the module to use symbolic algebra, so its behavior is too slow
    when using high order matrix.
    When only the output value is needed, this function is useful.
    """
    # judge whether clebsch gordan coefficient is zero or not
    if m1 + m2 != m3:
        return 0
    denominator = [j3+j1-j2, j3-j1+j2, j1+j2-j3, j3+m3, j3-m3]
    numerator = [j1+j2+j3+1, j1-m1, j1+m1, j2-m2, j2+m2]
    # It is obvious that "j1+j2-j3", "j3+m3" and "j3-m3" >= 0
    for i in denominator[:2]:
        if i < 0:
            return 0
    # It is obvious that "j1+j2+j3+1", "j1-m1", "j1+m1", "j2-m2" and "j2+m2" >= 0

    # determine the range of sum index
    max1 = j2 + j3 + m1
    max2 = j3 - j1 + j2
    max3 = j3 + m3
    min1 = m1 - j1
    min2 = j2 - j1 + m3
    vmax = min(max1, max2, max3)
    vmin = max(min1, min2, 0)

    dfactorial = 1
    for i in denominator:
        dfactorial *= factorial(i)
    nfactorial = 1
    for i in numerator:
        nfactorial *= factorial(i)
    coef = sqrt((2*j3+1)*dfactorial / nfactorial)
    vsum = 0
    for v in range(vmin, vmax+1):
        vsum += ((-1)**(v+j2+m2)*factorial(j2+j3+m1-v)*factorial(j1-m1+v)/
                 (factorial(v)*factorial(j3-j1+j2-v)*factorial(j3+m3-v)*factorial(v+j1-j2-m3)))
    return coef * vsum

def mkpair(evecs, mid):
    """Return the pair of eigen vector and key of hashed array

    Args:
        evecs (ndarray): eigen vector whose eigenvalue is 1
        mid (dict): return index from quantum numbers

    Returns:
        list: pair of quantum number set and eigen vector components
    """
    # make the indices list of evecs
    rlen = evecs.shape[1]
    ind = list(product(range(plen), range(rlen)))
    # make the indices list of zero components
    plc = np.where(np.isclose(evecs, 0))
    zeros = list(zip(plc[0], plc[1]))
    # get the indices list of non zero components from former lists
    for i in zeros:
        ind.remove(i)
    # sort ind to the same row number as evecs
    coef = [[i for i in ind if i[1] == j] for j in range(rlen)]
    # get the keys corresponding to the values, ndarray indices of evecs
    non_zero = [[k for i in coef[j] for k, v in mid.items() if v == i[0]] for j in range(rlen)]
    # link keys with eigen vectors
    co_val = [list(zip(non_zero[j], evecs[[mid[i] for i in non_zero[j]], j])) for j in range(rlen)]
    return co_val

if __name__ == "__main__":
    start = time.time()
    cg = clebsch_gordan(1, 2, 1, 1, -2, -1)
    end = time.time()
    elapsed = end - start
    print("{},{}(sec)".format(cg, elapsed))
    # get the l list
    print("Enter the azimuthal quantum number list of seed functions")
    lis = input("such as l1, l2 ,... , lp:")
    lis = lis.split(",")
    lis = [int(l) for l in lis]
    # calculate size of projection matrix and make it
    plen = 1
    for i in lis:
        plen *= 2 * i + 1
    pmat = np.zeros((plen, plen))
    # division to cases
    if len(lis) == 2 and lis[0] == lis[1]:
        id_list = list(product(range(-lis[0], lis[0]+1), range(-lis[1], lis[1]+1)))
        mid = {c: i for i, c in enumerate(id_list)}
        for cm in id_list:
            for rm in id_list:
                if cm[0] == -cm[1] and rm[0] == -rm[1]:
                    pmat[mid[cm], mid[rm]] = (-1)**(cm[1]-rm[1])/(2*lis[0]+1)
    elif len(lis) == 2 and lis[0] != lis[1]:
        # notification and end of system
        print("Projection matrix equals to zero matrix")
        sys.exit(0)
    if len(lis) == 3:
        id_list = list(product(range(-lis[0], lis[0]+1), range(-lis[1], lis[1]+1),
                               range(-lis[2], lis[2]+1)))
        mid = {c: i for i, c in enumerate(id_list)}
        for cm in id_list:
            for rm in id_list:
                c1 = clebsch_gordan(lis[0], lis[1], lis[2], cm[0], cm[1], -cm[2])
                c2 = clebsch_gordan(lis[0], lis[1], lis[2], rm[0], rm[1], -rm[2])
                pmat[mid[cm], mid[rm]] = (-1)**(cm[2]-rm[2])/(2*lis[2]+1)*c1*c2
    if len(lis) == 4:
        id_list = list(product(range(-lis[0], lis[0]+1), range(-lis[1], lis[1]+1),
                               range(-lis[2], lis[2]+1), range(-lis[3], lis[3]+1)))
        mid = {c: i for i, c in enumerate(id_list)}
        for cm in id_list:
            for rm in id_list:
                p = 0
                for l in range(abs(lis[0]-lis[1]), lis[0]+lis[1]):
                    c1 = clebsch_gordan(lis[0], lis[1], l, cm[0], cm[1], cm[0]+cm[1])
                    c2 = clebsch_gordan(lis[0], lis[1], l, rm[0], rm[1], rm[0]+rm[1])
                    c3 = clebsch_gordan(lis[2], l, lis[3], cm[2], cm[0]+cm[1], -cm[3])
                    c4 = clebsch_gordan(lis[2], l, lis[3], rm[2], rm[0]+rm[1], -rm[3])
                    p += (-1)**(rm[3]-cm[3])/(2*lis[3]+1)*c1*c2*c3*c4
                pmat[mid[cm], mid[rm]] = p
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
                        c1 = clebsch_gordan(lis[0], lis[1], l, cm[0], cm[1], cm[0]+cm[1])
                        c2 = clebsch_gordan(lis[0], lis[1], l, rm[0], rm[1], rm[0]+rm[1])
                        c3 = clebsch_gordan(lis[2], l, L, cm[2], cm[0]+cm[1], cm[0]+cm[1]+cm[2])
                        c4 = clebsch_gordan(lis[2], l, L, rm[2], rm[0]+rm[1], rm[0]+rm[1]+rm[2])
                        c5 = clebsch_gordan(lis[3], L, lis[4], cm[3], cm[0]+cm[1]+cm[2], -cm[4])
                        c6 = clebsch_gordan(lis[3], L, lis[4], rm[3], rm[0]+rm[1]+rm[2], -rm[4])
                        p += (-1)**(cm[4]-rm[4])*1/(2*lis[4]+1)*c1*c2*c3*c4*c5*c6
                pmat[mid[cm], mid[rm]] = p
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
                            c1 = clebsch_gordan(lis[0], lis[1], l, cm[0], cm[1], cm[0]+cm[1])
                            c2 = clebsch_gordan(lis[0], lis[1], l, rm[0], rm[1], rm[0]+rm[1])
                            c3 = clebsch_gordan(lis[2], l, L, cm[2], cm[0]+cm[1], cm[0]+cm[1]+cm[2])
                            c4 = clebsch_gordan(lis[2], l, L, rm[2], rm[0]+rm[1], rm[0]+rm[1]+rm[2])
                            c5 = clebsch_gordan(lis[3], L, S, cm[3], cm[0]+cm[1]+cm[2],
                                            cm[0]+cm[1]+cm[2]+cm[3])
                            c6 = clebsch_gordan(lis[3], L, S, rm[3], rm[0]+rm[1]+rm[2],
                                            rm[0]+rm[1]+rm[2]+rm[3])
                            c7 = clebsch_gordan(lis[4], S, lis[5], cm[4],
                                            cm[0]+cm[1]+cm[2]+cm[3], -cm[5])
                            c8 = clebsch_gordan(lis[4], S, lis[5], rm[4],
                                            rm[0]+rm[1]+rm[2]+rm[3], -rm[5])
                            p += (-1)**(cm[5]-rm[5])*1/(2*lis[5]+1)*c1*c2*c3*c4*c5*c6*c7*c8
                pmat[mid[cm], mid[rm]] = p
    # calculate eigenvalue and eigenvector
    # insert debugger
    # import pdb
    # pdb.set_trace()
    eig = np.linalg.eig(pmat)
    evecs = eig[1][:, np.isclose(eig[0], 1)]
    if np.any(evecs):
        co_val = mkpair(evecs, mid)
        print(co_val)
    else:
        print("Projection matrix has no eigen vector whose eigen value is 1.")
