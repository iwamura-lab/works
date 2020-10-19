"""
Program to calculate order parameters when selecting spherical harmonics as basis
"""
# set python interpreter(2 or 3 ?)
# !/usr/bin/python3
# -*- coding: UTF-8 -*-

# import standard modules
import os
import itertools
import pickle

# import modules including mathematical functions
import scipy.special as sp
import numpy as np

# import modules related to materialsProject
import pymatgen as mg

def calc_order_parameter(_structure, each_site, quantum, cut_off, params):
    """Calculate order parameter

    Args:
        _structure (Structure): crystal structure in POSCAR format
        each_site (Sites): the position of an atom
        _neighbors (list): neighboring atoms of each_site
        quantum (list): [l, m]
        cut_off (float): cut off radius
        params (dict): {"center": float, "height": float}

    Returns:
        float: order parameter
    """
    res = 0.0
    _neighbors = _structure.get_neighbors(each_site, cut_off)
    for neighbor in _neighbors:
        # transform cartesian coordinates into polar coordinates
        vec = neighbor.coords - each_site.coords
        r = np.linalg.norm(vec)
        theta = np.arccos(vec[2]/r)
        phi = np.arccos(vec[0]/(r*np.sin(theta)))
        if vec[1]/np.sin(theta) < 0:
            phi = -phi + 2 * np.pi
        # calculate radial function
        gauss = np.exp(- params["height"] * ((r - params["center"])**2))
        func_cut = 0.5 * (np.cos(np.pi * r/cut_off)+1)
        radial = gauss * func_cut
        res += radial * calc_sph_harm(quantum, phi, theta)
    return res.conjugate()


def calc_sph_harm(sph_indices, phi, theta):
    """Calculate spherical harmonics under given arguments

    Args:
        sph_indices (list): [l, m]
        phi (float): azimuthal angle
        theta (float): polar angle

    Returns:
        float: spherical harmonics

    Since the default definition of quantum numbers and angles is not natural
    to physicists, there was urgent necessity to modify.
    """

    return sp.sph_harm(sph_indices[1], sph_indices[0], phi, theta)


def calc_opl(poscar, lmax, cut_off, params):
    """Calculate multi list of order parameters in using spherical harmonics as basis functions

    Args:
        poscar (POSCAR): the path of POSCAR file(information of atomic positions)
        lmax (int): maximum azimuthal quantum number
        cut_off (float): cut off radius
        params (dict): parameters of radial function {center: float, height: float}

    Returns:
        list: order parameter centered at atoms in unit cell
        float: cut_off radius
    """

    # get information of atomic positions by reading POSCAR file
    _structure = mg.Structure.from_str(open(poscar).read(), fmt="poscar")
    atom_sites = _structure.sites
    # make the iterator which returns all combinations of atom_sites
    pairs = itertools.combinations(atom_sites, 2)
    for pair in pairs:
        dist = pair[0].distance(pair[1])
        if dist < cut_off:
            cut_off = dist
    cut_off = (cut_off + 0.5) * 2

    # prepare data structure
    res = []

    # get information of neighboring atoms and calculate order parameters
    for each_site in atom_sites:
        # prepare data structure
        al = []
        for l in range(lmax+1):
            a = []
            for m in range(-l, l+1):
                a.append(calc_order_parameter(_structure, each_site, [l, m], cut_off, params))
            # make the list of order parameters
            al.append(a)
        res.append(al)
    return res, cut_off

if __name__ == "__main__":
    # get the path of files included in dataset
    poscars = os.listdir("dataset")
    # set some values to the parameters
    lmax = 3
    rpar = {"center": 0.0, "height": 1.0}
    for cnt, path in enumerate(poscars):
        # reset cut-off radius
        cr = 100
        opl, cut_off = calc_opl("dataset/"+path, lmax, cr, rpar)
        opl.append(cut_off)
        pickle.dump(opl, open("results/lmax3/"+path+".dump", "wb"))
        print(cnt+1)
