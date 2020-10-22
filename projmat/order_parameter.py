"""
Program to calculate order parameters when selecting spherical harmonics as basis
"""
# set python interpreter(2 or 3 ?)
# !/usr/bin/python3
# -*- coding: UTF-8 -*-

# import standard modules
import os
from math import sqrt, factorial
import itertools
import shelve
import pickle

# import modules including mathematical functions
import scipy.special as sp
from scipy.integrate import tplquad
import numpy as np

# import modules related to materialsProject
import pymatgen as mg

def sph_harm_R(phi, theta, l, m):
    """Return real part of spherical harmonics

    Args:
        phi (float): azimuthal angle
        theta (float): polar angle
        l (int): azimuthal quantum number
        m (int): magnetic quantum number

    Returns:
        float: real part of sph_harm(real number)
    """

    coef = sqrt(((2*l+1)*factorial(l-m))/(4*np.pi*factorial(l+m)))
    return coef * np.cos(m * phi) * sp.lpmv(m, l, np.cos(theta))

def sph_harm_I(phi, theta, l, m):
    """Return imaginary part of spherical harmonics

    Args:
        phi (float): azimuthal angle
        theta (float): polar angle
        l (int): azimuthal quantum number
        m (int): magnetic quantum number

    Returns:
        float: imaginary part of sph_harm(real number)
    """

    coef = sqrt(((2*l+1)*factorial(l-m))/(4*np.pi*factorial(l+m)))
    return coef * np.sin(m * phi) * sp.lpmv(m, l, np.cos(theta))

def gaussian(x, mu, sigma):
    """Calculate gaussian function

    Args:
        x (float): changable variable
        mu (float): central position
        sigma (float): standard deviation of data

    Returns:
        float: result of calculation
    """

    return np.exp(-(x - mu)**2/(2*sigma**2))

def a_int_R(r, theta, phi, qnum, cut_off, mu):
    """Return real part of integrand to use scipy.integrate.tplquad

    Args:
        r (float): radius
        theta (float): polar angle
        phi (float): azimuthal angle
        qnum (tuple): tuple of quantum numbers
        cut_off (float): cut_off radius
        mu (list): cartesian coordinates of neighboring atom

    Returns:
        float: integrand
    """

    # set a value to sigma
    sigma = 8.5 * 10**(-3)
    # sigma = 1
    # calculate cartesian coordinates from polar coordinates
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    # calculate density function
    coef = 1/(sqrt(2*np.pi)*sigma)**3
    fx = gaussian(x, mu[0], sigma)
    fy = gaussian(y, mu[1], sigma)
    fz = gaussian(z, mu[2], sigma)
    # calculate radial function and spherical harmonics
    gauss = np.exp(- r**2)
    func_cut = 0.5 * (np.cos(np.pi * r/cut_off)+1)
    radial = gauss * func_cut
    sph = sph_harm_R(phi, theta, qnum[0], qnum[1]) * r**2 * np.sin(theta)
    return coef * fx * fy * fz * radial * sph

def a_int_I(r, theta, phi, qnum, cut_off, mu):
    """Return real part of integrand to use scipy.integrate.tplquad

    Args:
        r (float): radius
        theta (float): polar angle
        phi (float): azimuthal angle
        qnum (tuple): tuple of quantum numbers
        cut_off (float): cut_off radius
        mu (list): cartesian coordinates of neighboring atom

    Returns:
        float: integrand
    """

    # set a value to sigma
    sigma = 8.5 * 10**(-3)
    # sigma = 1
    # calculate cartesian coordinates from polar coordinates
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    # calculate density function
    coef = 1/(sqrt(2*np.pi)*sigma)**3
    fx = gaussian(x, mu[0], sigma)
    fy = gaussian(y, mu[1], sigma)
    fz = gaussian(z, mu[2], sigma)
    # calculate radial function and spherical harmonics
    gauss = np.exp(- r**2)
    func_cut = 0.5 * (np.cos(np.pi * r/cut_off)+1)
    radial = gauss * func_cut
    sph = sph_harm_I(phi, theta, qnum[0], qnum[1]) * r**2 * np.sin(theta)
    return coef * fx * fy * fz * radial * sph

def calc_order_parameter(vec, quantum, cut_off, params):
    """Calculate order parameter

    Args:
        _structure (Structure): crystal structure in POSCAR format
        each_site (Sites): the position of an atom
        _neighbors (list): neighboring atoms of each_site
        quantum (tuple): (l, m)
        cut_off (float): cut off radius
        params (float): the central position of gauss function

    Returns:
        float: order parameter
    """

    res = 0.0
    # transform cartesian coordinates into polar coordinates
    r = np.linalg.norm(vec)
    theta = np.arccos(vec[2]/r)
    phi = np.arccos(vec[0]/(r*np.sin(theta)))
    if vec[1]/np.sin(theta) < 0:
        phi = -phi + 2 * np.pi
    # calculate radial function
    gauss = np.exp(- (r - params)**2)
    func_cut = 0.5 * (np.cos(np.pi * r/cut_off)+1)
    radial = gauss * func_cut
    res += radial * calc_sph_harm(quantum, phi, theta)
    return res.conjugate()

def calc_sph_harm(sph_indices, phi, theta):
    """Calculate spherical harmonics under given arguments

    Args:
        sph_indices (tuple): (l, m)
        phi (float): azimuthal angle
        theta (float): polar angle

    Returns:
        float: spherical harmonics

    Since the default definition of quantum numbers and angles is not natural
    to physicists, there was urgent necessity to modify.
    """

    return sp.sph_harm(sph_indices[1], sph_indices[0], phi, theta)

def calc_opl(vec, ref_dict, cut_off, nmax):
    """Calculate multi list of order parameters in using spherical harmonics as basis functions

    Args:
        vec (float): the distance between a site and neighboring atoms
        ref_dict (dict): receive pair of quantum number (l, m) and return index(int)
        cut_off (float): cut off radius
        nmax (dict): maximum of center position in gauss function

    Returns:
        ndarray: basis function values at the position of a neighboring atom
    """

    # prepare data structure
    res = np.zeros((nmax+1, len(ref_dict)), dtype=np.complex)
    # make the iterator
    seq = itertools.product([i for i in range(nmax+1)], list(ref_dict))
    for nlm in seq:
        res[nlm[0], ref_dict[nlm[1]]] = calc_order_parameter(vec, nlm[1], cut_off, nlm[0])
    return res

def calc_order_parameter2(poscar, ref_dict, cut_off):
    """Calculate order parameter when taking approximation of rho(i)

    Args:
        poscar (poscar): the path of POSCAR file
        ref_dict (dict): hash dictionary of l and m
        cut_off (flaot): cut_off radius

    Returns:
        ndarray: ndarray of order_parameters
    """
    _structure = mg.Structure.from_str(open(poscar).read(), fmt="poscar")
    atom_sites = _structure.sites
    for i, each_site in enumerate(atom_sites):
        _neighbors = _structure.get_neighbors(each_site, cut_off)
        order_parameters = np.array([])
        for lm in list(ref_dict):
            a_R = a_I = 0.0
            for oposite in _neighbors:
                vec = oposite.coords - each_site.coords
                a_R += tplquad(a_int_R, 0, 2 * np.pi, lambda phi: 0, lambda phi: np.pi,
                               lambda phi, theta: 0, lambda phi, theta: cut_off,
                               (lm, cut_off, vec))[0]
                if lm[1] != 0:
                    a_I += tplquad(a_int_I, 0, 2 * np.pi, lambda phi: 0, lambda phi: np.pi,
                                   lambda phi, theta: 0, lambda phi, theta: cut_off,
                                   (lm, cut_off, vec))[0]
            order_parameters = np.append(order_parameters, complex(a_R, a_I))
        if i == 0:
            res = order_parameters
        else:
            res = np.vstack([res, order_parameters])
    return res

if __name__ == "__main__":
    # get the path of files included in dataset
    poscars = os.listdir("dataset")
    # set or get initial values for the parameters
    lmax = 2
    nmax = 5
    lm_pair = [(l, m) for l in range(lmax+1) for m in range(-l, l+1)]
    ref_dict = {key: i for i, key in enumerate(lm_pair)}
    cr_values = shelve.open("results/datacheck/nearest.db")
    scale = 2
    # decide which mode will be used
    mode = 1
    if mode == 1:
        for cnt, path in enumerate(poscars):
            # calculate cut_off radius
            cut_off = scale * cr_values[path]
            # get information of atomic positions by reading POSCAR file
            _structure = mg.Structure.from_str(open("dataset/"+path).read(), fmt="poscar")
            atom_sites = _structure.sites
            # get information of neighboring atoms and calculate order parameters
            for i, each_site in enumerate(atom_sites):
                res = np.zeros((nmax+1, len(ref_dict)), dtype=np.complex)
                for oposite in _structure.get_neighbors(each_site, cut_off):
                    vec = oposite.coords - each_site.coords
                    res += calc_opl(vec, ref_dict, cut_off, nmax)
                pickle.dump(res, open("results/delta/lmax2"+path+"_"+str(i+1)+".dump", "wb"))
            print(cnt+1)
    elif mode == 2:
        for cnt, path in enumerate(poscars):
            res = calc_order_parameter2("dataset/"+path, ref_dict, cut_off)
            pickle.dump(res, open("results/normal_dis/order_parameters/lmax2/"+path+".dump", "wb"))
            print(cnt+1)
        pickle.dump(ref_dict, open("results/normal_dis/order_paramters/ref_dict.dump", "wb"))
