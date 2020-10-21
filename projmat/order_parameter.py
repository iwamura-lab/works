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
    lmax = 3
    lm_pair = [(l, m) for l in range(lmax+1) for m in range(-l, l+1)]
    ref_dict = {key: i for i, key in enumerate(lm_pair)}
    cr_values = shelve.open("results/datacheck/nearest.db")
    scale = 2
    # decide which mode will be used
    mode = 2
    if mode == 1:
        for cnt, path in enumerate(poscars):
            # reset cut-off radius and parameters
            cr = 100
            rpar = {"center": 0.0, "height": 1.0}
            opl, cut_off = calc_opl("dataset/"+path, lmax, cr, rpar)
            opl.append(cut_off)
            pickle.dump(opl, open("results/lmax3/"+path+".dump", "wb"))
            print(cnt+1)
    elif mode == 2:
        for cnt, path in enumerate(poscars):
            res = calc_order_parameter2("dataset/"+path, ref_dict, scale * cr_values[path])
            print(cnt+1)
