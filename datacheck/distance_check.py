"""
Program to examine whether it is possible or not to measure the nearest atomic
distance just by one loop regarding structure datas
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

def ndist_error(poscars):
    """Return the difference between nearest distance in one loop and that of all combi

    Args:
        poscars (list): list of file paths

    Returns:
        ndarray: ndarray of difference when changing the data file
    """
    res = np.array([])
    for cnt, path in enumerate(poscars):
        _structure = mg.Structure.from_str(open("dataset/"+path).read(), fmt="poscar")
        atom_sites = _structure.sites
        one_lp = all_lp = 100
        # when executing just one loop
        for opsite in atom_sites[1:]:
            dist = atom_sites[0].distance(opsite)
            if dist < one_lp:
                one_lp = dist
        # when taking all combinations of atom_sites
        pairs = itertools.combinations(atom_sites, 2)
        for pair in pairs:
            dist = pair[0].distance(pair[1])
            if dist < all_lp:
                all_lp = dist
        res = np.append(res, (one_lp - all_lp))
        print(cnt+1)
    return res


if __name__ == "__main__":
    # get the path of files included in dataset
    poscars = os.listdir("dataset")
    res = ndist_error(poscars)
    pickle.dump(res, open("results/datacheck/1vsAll.dump", "wb"))
