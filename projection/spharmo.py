'''
This script calcs spherical harmonics x radius function from POSCAR file
'''


# !/usr/bin/env/python
# -*- coding: UTF-8 -*-

# Import modules
import numpy as np
import math
import pandas as pd
import scipy.special as sp


def calc_product_sph_radius(poscar, sph_basis, cut_off, params,
                            radius_type='gaussian', cut_off_func='cos',
                            csv_path=None, gamma=1):
    '''
    Calcluate spherical harmonics x radius function value from POSCAR file
    -----------------------------------------------------------------------
    input
    poscar : pymatge.io.vasp.Poscar class
        The poscar you want to calc
    sph_basis : list or np array like (The length must be 2)
        The l and m of spherical harmonics value
        Ex) [2, 1] -----------> l = 2 and m = 1
    cut_off : float
        The cut off radius
    params : dict
        The dict of gaussian function
        Ex) {'center_param' : 3, 'height_param' : 1}
    radius_type : str (default is gaussian)
        The radius function type you use
        gaussian : Only using gaussian function
        gaussian_weighed : weighing gaussian values by the differences of kai
    cut_off_func : str (default is cos)
        The cut off function type you use
    ------------------------------------------------------------------------
    '''
    result = []
    _structure = poscar.structure
    atom_sites = _structure.sites

    for each_site in atom_sites:
        _neighbors = _structure.get_neighbors(each_site, cut_off)
        # Get every neighbors angle data
        _angles = [get_angle_in_sph_coords(each_site.coords,
                                           each_neighbor[0].coords)
                   for each_neighbor in _neighbors]

        # Get spherical harmonics value
        if sph_basis is None:
            pass
        else:
            sph_vals = [calc_sph_value(sph_basis, angle[0], angle[1])
                        for angle in _angles]

        # Get every neighbrs distance data
        _distance = [each_neighbor[1] for each_neighbor in _neighbors]

        _species_name = [each_neighbor[0].species_string
                         for each_neighbor in _neighbors]
        center = params['center_param']
        height = params['height_param']

        # Calc radius function
        if radius_type == 'gaussian':
            radius_val = [calc_gaussian(dist, cut_off, center, height,
                                        cut_off_func)
                          for dist in _distance]

        if radius_type == 'gaussian_weighed':
            tmp_radius = [calc_gaussian(dist, cut_off, center, height,
                                        cut_off_func)
                          for dist in _distance]

            # Read csv file which has atomic data
            df = pd.read_csv(csv_path)

            # Get center atom's pauling kai value
            center_kai = np.float(df[df['Symbol'] == each_site.species_string]
                                  ['kai(Pauling)'].values)

            # Get neighbors kai value
            neighbors_kai = [np.float(df[df['Symbol'] == i]
                                      ['kai(Pauling)'].values)
                             for i in _species_name]

            # Calc the difference of center_kai and neighbors_kai
            diff_kai = np.array(neighbors_kai) - center_kai

            # weighed value
            weighed_kai_val = [(gamma * ((each_kai) ** 2)) + 1
                               for each_kai in diff_kai]

            # Calc product of them
            radius_val = np.array(tmp_radius) * np.array(weighed_kai_val)

        # If sph_basis is set to None, return only radius val
        if sph_basis is None:
            each_result = sum(radius_val)
        else:
            # calc product of them
            each_result = np.dot(np.array(sph_vals).reshape(1, -1),
                                 np.array(radius_val).reshape(-1, 1))
        result.append(each_result)

    return np.array(result).reshape(-1, 1)


# Define the radius function
def calc_gaussian(radius, cut_off,
                  center_param, height_param, cut_off_func='cos'):
    '''
    Calculates radius function of atomby Guassian Function
    input
    ------------------------------------------------------------------------
    radius : float
        The radius between a atom and another atom which you focus on
    cut_off : float
        The cut off radius
    center_param : float
        The hyper parameter which decides center of Gaussian function
    height_param : float
        The hyper paramter which decides the height of Gaussian function
    cutoff_func : str
        The type of cut off function (default is cos)
        (afterwards other type of function to be implemented
    -------------------------------------------------------------------------
    '''
    gauss_rad = np.exp(- height_param * ((radius - center_param)**2))
    if cut_off_func == 'cos':
        cutoff_effect = 1/2 * (math.cos(math.pi * (radius/cut_off)) + 1)
    else:
        raise Exception('Cut off function must be cos!!')
    result = gauss_rad * cutoff_effect
    return result


def get_angle_in_sph_coords(origin_coords, target_coords, distance=None):
    '''
    input
    ------------------------------------------------------------------------
    origin_coords : np array like (The shape is (1, 3))
    target_coords : np.array like (The shape is (1, 3))
        The order must be like [x, y, z]
    distance : float
        The distance of two atoms
    ------------------------------------------------------------------------
    '''

    # raise error message
    if len(origin_coords) != 3:
        raise Exception('The coords info must be 3 dimention!!')
    else:
        trans_coords = target_coords - origin_coords
        # Get distance data
        if distance is None:
            distance = np.linalg.norm(trans_coords)
        else:
            distance = distance

        # theta is z-axis angle (polar angle)
        theta = math.acos(trans_coords[-1]/distance)

        # if x, y are 0, phi can not be defined
        # so return phi = 0
        # phi value is not related to spherical harmonics if theta is 0
        if np.allclose(trans_coords[0:-1], [0, 0], atol=1e-3):
            phi = 0

        else:
            # phi is x-axis angle (azimuthal angle)
            phi = math.atan2(trans_coords[1], trans_coords[0])

            # phi should be 0<phi<2pi
            if phi < 0:
                phi += 2 * math.pi

        return theta, phi


def calc_sph_value(sph_basis, theta, phi):
    # basis order is reverse in my program
    # This should be improved
    # Also theta and phi definition is reverse too
    val = sp.sph_harm(sph_basis[1], sph_basis[0], phi, theta)
    return np.conjugate(val)


if __name__ == '__main__':
    from pymatgen.io.vasp import Poscar
    POSCAR = Poscar.from_file('/home/ryuhei/poscar_data/cohesive/descriptors/'
                              'A2X3/2085_A2XY2_Bi2Te3/Al-O/POSCAR')
    val = calc_product_sph_radius(POSCAR, [2, 0], 6, [0, 1],
                                  'gaussian_weighed',
                                  csv_path='/media/sf_Share/'
                                           'atomic_data_20160603.csv')
    print(val)
