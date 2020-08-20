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


def calc_order_parameter(poscar):
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
                          {center: float, height: float}
    -----------------------------------------------------------------------------
    """
    _structure = mg.Structure.from_file(poscar)
    return _structure
