'''
Practice of python coding and software development by using github
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
