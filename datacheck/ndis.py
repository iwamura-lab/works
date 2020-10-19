"""
Program to get the data of nearest atomic distance regarding structure data
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

def ndis(poscars):
    """Return the nearest atomic distance of all pairs

    Args:
        poscars (list): list of file paths

    Returns:
        ndarray: ndarray of the nearest atomic distances
    """
    res = np.array([])
    for cnt, path in enumerate(poscars):
        # get the information of structure and sites from data file
        _structure = mg.Structure.from_str(open("dataset/"+path).read(), fmt="poscar")
        atom_sites = _structure.sites
        # take all combinations of atom_sites
        pairs = itertools.combinations(atom_sites, 2)
        ndis = 100
        for pair in pairs:
            dist = pair[0].distance(pair[1])
            if dist < ndis:
                ndis = dist
        res = np.append(res, ndis)
        print(cnt+1)
    return res


if __name__ == "__main__":
    # get the path of files included in dataset
    poscars = os.listdir("dataset")
    res = ndis(poscars)
    pickle.dump(res, open("results/datacheck/1vsAll.dump", "wb"))
