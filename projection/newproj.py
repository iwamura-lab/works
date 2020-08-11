'''
Module for computing invariants of O(3) group systematically
Using Projection Operator Method
This Module assumes that use basis function
[spherical harmonics] and [Gaussian Hermite Moment]
each representaion matrixs is [Wigner D-Matrix] and [Euler Rotation Matrix]
'''

# !/usr/bin/env/python
# -*- coding: UTF-8 -*-

# Import modules
import numpy as np
from sympy import Symbol
from sympy.physics.quantum.spin import WignerD
from sympy import cos
from sympy import sin
from sympy import exp
from sympy import I
from sympy import integrate
from sympy import symbols
import itertools
import math
from joblib import Parallel
from joblib import delayed
from scipy.integrate import quad

class NewProjMatrix():
    def __init__(self):
        self.invariants = None
        self.eigs = None    # This is eigen values
        self.proj_matrix = None   # This is projection matrix

    def set_basis(self, l_vals, exponents):
        '''
        set basis functions
        input
        -----------------------------------------------------
        l_vals : l values of spherical harmonics
        exponents : exponent values of Gaussian Hermite Moment
        if you want to use only one of them, please set None
        '''
        # Set values of basis functions
        self.l_vals = l_vals
        self.exponens = exponents

        # If l_vals is None, unuse spherical harmonics
        if l_vals is None:
            self.sph_basis = None
            self.ghm_basis = get_ghm_basis(exponents)
            self.basis = self.ghm_basis

        # If exponents is None, unuse Gaussian Hermite Moment
        elif exponents is None:
            self.sph_basis = get_sph_basis(l_vals)
            self.ghm_basis = None
            self.basis = self.sph_basis

        # Calc kronecker product of basis functions
        else:
            self.sph_basis = get_sph_basis(l_vals)
            self.ghm_basis = get_ghm_basis(exponents)
            basis = list(itertools.product(self.sph_basis, self.ghm_basis))
            # Set value
            self.basis = basis

    def calc_rep(self):
        '''
        Calculate representaion matrix of basis functions
        '''
        # if sph_basis is None, unuse spherical harmonics
        if self.sph_basis is None:
            self.rep_matrix = calc_ghm_kron_rep(self.exponens)
        # if ghm_basis is None, unuse ghm basis
        elif self.ghm_basis is None:
            self.rep_matrix = calc_sph_kron_rep(self.l_vals)
        # else, use both of them
        else:
            sph_matrix = calc_sph_kron_rep(self.l_vals)
            ghm_matrix = calc_ghm_kron_rep(self.exponens)
            self.rep_matrix = np.kron(sph_matrix, ghm_matrix)

    def projection(self):
        self.proj_matrix = calc_proj_matrix(self.rep_matrix)

    def calc_invs(self):
        '''
        Calulate invariants by using eigen value method
        '''
        matrix = self.proj_matrix
        # Solve eigen value problem
        eigs = np.linalg.eig(matrix)
        # Select only eigen vecor, whose eigen value == 1
        eig_arr = eigs[1][:, np.isclose(eigs[0], 1)]

        self.eigs = eig_arr


def get_sph_basis(l_vals):
    '''
    Calculate basis functions of spherical harmonics
    '''
    _basis_li = []
    for each_l in l_vals:
        m_vals = range(-each_l, each_l + 1)
        _lst = [[each_l, m] for m in m_vals]
        _basis_li.append(_lst)
        # Calc kronecker product of basis functions
        sph_basis = list(itertools.product(*_basis_li))
    return sph_basis


def get_ghm_basis(exponents):
    '''
    Calculate basis function of Gaussian Hermite Moment
    '''
    _lst = [list(itertools.product([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   repeat=expo)) for expo in exponents]
    _rep = []
    for each in _lst:
        each_rep = [list(sum(np.array(a_file))) for a_file in each]
        _rep.append(each_rep)

    knor_li = list(itertools.product(*_rep))
    return knor_li


def calc_sph_kron_rep(l_vals):
    '''
    Calculate representaion matrix of kronecker product of spherical harmonics
    input
    ----------------------------------------------------------
    l_vals : list or numpy array like
        The list of l values to calc kronecker product
        Ex)[1, 1]------------> calc l=1 x l=1
    '''
    _reps = [calc_each_sph_rep(l) for l in l_vals]
    for i, arr in enumerate(_reps):
        if i == 0:
            matrix = arr
            new_matrix = arr
        else:
            new_matrix = np.kron(matrix, arr)
            matrix = new_matrix
    return new_matrix


def calc_each_sph_rep(l):
    '''
    Calculate representaion matrix of spherical harmonics
    inpt
    ----------------------------------------------------------
    l : int
        The l value to calc representaion matrix
    '''
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    # a is rotation angle of z-axis
    # b is rotation angle of x'-axis
    # c is rotation angle of z'-axis
    _mat = np.zeros((1, 2*l + 1))
    for i, m in enumerate(range(-l, l + 1)):
        vec = [WignerD(l, m, mp, a, b, c) for mp in range(-l, l + 1)]
        _mat = np.vstack((_mat, np.array(vec).reshape(1, -1)))
    matrix = _mat[1:]
    return matrix


def calc_ghm_kron_rep(exp_lst):
    '''
    Calculate representaion matrix of kronecker product of
    Gaussian Hermite Moment
    input
    ----------------------------------------------------------
    exp_lst : list or numpy array like
        The exponent value list to calc
        Ex)[2,2,2]------->kronecker product of p+q+r=2 **3
    ----------------------------------------------------------
    '''
    _reps = [calc_each_ghm_rep(_exp) for _exp in exp_lst]
    for i, arr in enumerate(_reps):
        if i == 0:
            matrix = arr
            new_matrix = arr
        else:
            new_matrix = np.kron(matrix, arr)
            matrix = new_matrix
    return new_matrix


def calc_each_ghm_rep(exponent):
    '''
    Calculate representaion matrix of Gaussian Hermite Moment
    input
    -----------------------------------------------------------
    exponent : int
        The value of Moment order to calc matrix
    -----------------------------------------------------------
    '''
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    # a is rotation angle of z-axis
    # b is rotation angle of x'-axis
    # c is rotation angle of z'-axis
    # r is rotaion matrix
    r = np.array(([cos(c)*cos(a)-cos(b)*sin(a)*sin(c),
                   cos(c)*sin(a)+cos(b)*cos(a)*sin(c),
                   sin(c)*sin(b), -sin(c)*cos(a)-cos(b)*sin(a)*cos(c),
                   -sin(c)*sin(a)+cos(b)*cos(a)*cos(c), cos(c)*sin(b),
                   sin(b)*sin(a), -sin(b)*cos(a), cos(b)])).reshape(3, 3)

    for i in range(exponent):
        if i == 0:
            matrix = r
            new_matrix = r
        else:
            new_matrix = np.kron(matrix, r)
            matrix = new_matrix
    return new_matrix


def calc_proj_matrix(matrix):
    '''
    Calculate projection matrix of basis function
    Reduce computation cost by omitting same basis functions' integration
    '''
    # Create zero matrix which is result
    proj_mat = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.complex)

    # Get info of element of representaion matrix
    # To reduce computing cost, calc only one element if one appears many times
    _lst = list(map(list, matrix))
    _flaten_lst = [each for a_file in _lst for each in a_file]

    # Get list of values which is uniq
    _uniq_lst = list(set(_flaten_lst))
    print('Calclation will be done for ' + str(len(_uniq_lst)))

    # Calulate element of Projection matrix
    # Doing parallel calculation
    _cals = Parallel(n_jobs=-1, verbose=80)([delayed(integrate_element)(ele, i)
                                             for i, ele
                                             in enumerate(_uniq_lst)])

    # Sort result of calculation by index
    _cals.sort(key=lambda x: x[1])
    _vals = [t[0] for t in _cals]

    # Rewite result matrix
    for row, col in itertools.product(range(matrix.shape[0]),
                                      range(matrix.shape[1])):
        ele = matrix[row, col]
        ind = _uniq_lst.index(ele)
        val = _vals[ind]
        proj_mat[row, col] = val
    return proj_mat


def integrate_element(element, i=None):
    '''
    Calculate element of Projection matrix

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    # Integrate a value
    val1 = integrate(element, (a, 0, 2 * math.pi)).evalf()
    # Doing integrate
    val1 = val1.doit()
    # Integrate c value
    val2 = integrate(val1, (c, 0, 2 * math.pi)).evalf()
    val2 = val2.doit()
    # Integrate b value
    val3 = integrate(val2 * sin(b), (b, 0, math.pi), risch=False).evalf()

    D
    # Doing integration if val3 is sympy.core.mul.Mul type
    val3 = val3.doit()
    '''

    a, b, c = symbols('a b c')
    # val3 = integrate(element * sin(b), (a, 0, 2 * math.pi),
    #                  (b, 0, math.pi)).evalf()
    # integrate around a
    _tmp1 = integrate(element, (a, 0, 2 * math.pi)).evalf()
    _tmp1 = _tmp1.doit()

    _tmp2 = integrate(_tmp1, (c, 0, 2 * math.pi))
    _tmp2 = _tmp2.doit()

    # Integrate by quad method
    # Separate real and imag
    real, imag = _tmp2.as_real_imag()
    real = real * sin(b)
    imag = imag * sin(b)
    # tmp_fuc = val3.doit()
    # real, imag = tmp_fuc.as_real_imag()

    def __real_func(b_val):
        res = real.subs(b, b_val)
        return res

    def __imag_func(b_val):
        res = imag.subs(b, b_val)
        return res

    float_val = quad(__real_func, 0, 2 * math.pi)[0]
    complex_val = quad(__imag_func, 0, 2 * math.pi)[0]

    print(float_val)
    print(complex_val)

    return np.complex(float_val + complex_val * I)
    '''
    # cast to complex value
    complex_val = complex(val3)
    result = complex_val/(8 * math.pi ** 2)
    if i is None:
        return result
    else:
        return result, i
    '''

if __name__ == '__main__':
    INS = NewProjMatrix()
    INS.set_basis([2], [2])
    INS.calc_rep()
    RES = integrate_element(INS.rep_matrix[9, 13])
    print(RES)
