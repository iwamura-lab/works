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

    sph_indices.reverse()
    return sp.sph_harm(sph_indices[0], sph_indices[1], phi, theta)


def calc_order_parameter(poscar, l, m, cut_off, params):
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
    # change data structure of quantum numbers
    quantum = [l, m]
    results = []

    # get information of neighboring atoms and calculate order parameters
    for each_site in atom_sites:
        # prepare data structures
        a = 0
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
            a += radial * calc_sph_harm(quantum, phi, theta)
        # make the list of order parameters
        results.append(a)
    return np.conj(results)

if __name__ == "__main__":
    n1 = int(input("Enter the azimuthal quantum number.:"))
    n2 = int(input("Enter the magnetic quantum number.:"))
    cr = float(input("Enter the cut-off radius.:"))
    rpar = {"center": 0, "height": 0}
    rpar["center"] = float(input("Enter the central position of radial function.:"))
    rpar["height"] = float(input("Enter the height of radial function.:"))
    print(calc_order_parameter("POSCAR", n1, n2, cr, rpar))
        