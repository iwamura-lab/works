"""
Program to calculate the standard deviation of 1st neighboring atomic distance
regarding structure datas
"""
# set python interpreter(2 or 3 ?)
# !/usr/bin/python3
# -*- coding: UTF-8 -*-

# import standard modules
import os
import itertools
import pickle

# import modules including mathematical functions
import numpy as np

# import modules related to materialsProject
import pymatgen as mg

def calc_std_1stN(poscars):
    """Calculate the standard deviation of 1st neighboring atomic distance

    Args:
        poscars (list): list of file paths

    Returns:
        ndarray: ndarray of difference when changing the data file
    """
    res = np.array([])
    for cnt, path in enumerate(poscars):
        _structure = mg.Structure.from_str(open("dataset/"+path).read(), fmt="poscar")
        atom_sites = _structure.sites
        pairs = itertools.combinations(atom_sites, 2)
        nst_lis = [pair[0].distance(pair[1]) for pair in pairs]
        res = np.append(res, max(nst_lis)-min(nst_lis))
        print(cnt+1)
    return res

if __name__ == "__main__":
    # get the path of files included in dataset
    poscars = os.listdir("dataset")
    res = calc_std_1stN(poscars)
    pickle.dump(res, open("results/datacheck/std_of_1stN.dump", "wb"))
