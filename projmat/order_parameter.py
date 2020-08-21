'''
Practice of python coding and software development by using Github
'''
# set python interpreter(2 or 3 ?)
# !/usr/bin/python3
# -*- coding: UTF-8 -*-

# import modules including mathematical functions
import scipy.special as sp
import numpy as np

# import modules related to materialsProject
import pymatgen as mg

def calc_sph_harm(sph_indices, phi, theta):
    """
    Docstring:
    calc_sph_harm(sph_indices, phi, theta)

    Calculate spherical harmonics under given arguments
    (sph_indices: [l, m], phi: azimuthal angle, theta: polar angle).
    Since the default definition of quantum numbers and angles is not natural
    to physicists, there was urgent necessity to modify.
    """

    sph_indices = sph_indices.reverse()
    return sp.sph_harm(sph_indices[0], sph_indices[1], phi, theta)


def calc_order_parameter(poscar, cut_off):
    """
    Docstring:
    calc_order_parameter(poscar, l, m, cut_off, params)

    Calculate order parameters when using spherical harmonics as basis functions.
    -----------------------------------------------------------------------------
    Input
        poscar (POSCAR) : POSCAR file(information of atomic positions)
        l      (int)    : azimuthal quantum number
        m      (int)    : magnetic quantum number
        cut_off(float)  : cut off radius
        params (dict)   : parameters of radial function
                          {center: float,  height: float}
    -----------------------------------------------------------------------------
    """
    # get information of atomic positions by reading POSCAR file
    _structure = mg.Structure.from_file(poscar)
    atom_sites = _structure.sites
    pos = []

    # get information of neighboring atoms and calculate order parameters
    for each_site in atom_sites:
        my_coords = []
        _neighbors = _structure.get_neighbors(each_site, cut_off)
        for neighbor in _neighbors:
            vec = neighbor.coords - each_site.coords
            r = np.linalg.norm(vec)
            theta = np.arccos(vec[2]/r)
            phi = np.arccos(vec[0]/(r*np.sin(theta)))
            if vec[1]/np.sin(theta) < 0:
                phi = -phi + 2 * np.pi
            my_coords.append([r, theta, phi])
        pos.append(my_coords)

    return pos
