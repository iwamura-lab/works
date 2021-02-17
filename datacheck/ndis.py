#!/usr/bin/env python
"""
Program to get the data of nearest atomic distance regarding structure data
"""

# import standard modules
import os
import itertools
import shelve

# import modules related to materialsProject
import pymatgen as mg

def ndis(poscars):
    """Return the nearest atomic distance of all pairs

    Args:
        poscars (list): list of file paths

    Returns:
        ndarray: ndarray of the nearest atomic distances
    """
    res = dict()
    for cnt, path in enumerate(poscars):
        # get the information of structure and sites from data file
        _structure = mg.Structure.from_str(open("dataset/"+path).read(), fmt="poscar")
        atom_sites = _structure.sites
        # take all combinations of atom_sites
        pairs = itertools.combinations(atom_sites, 2)
        # initializer
        ndis = 100
        for pair in pairs:
            dist = pair[0].distance(pair[1])
            if dist < ndis:
                ndis = dist
        res[path] = ndis
        print(cnt+1)
    return res


if __name__ == "__main__":
    # get the path of files included in dataset
    poscars = os.listdir("dataset")
    res = ndis(poscars)
    data_base = shelve.open("results/datacheck/nearest.db")
    data_base.update(res)
    data_base.close()
