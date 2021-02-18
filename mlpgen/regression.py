#!/usr/bin/env python
"""
Program to execute the regression of mlp about paramagnetic FCC Fe
"""

# import modules to test
import time

# import standard modules
import copy
import argparse
#import tqdm
import numpy as np

# from mlptools import some modules
from mlptools.common.fileio import InputParams
from mlptools.common.readvasp import Vasprun
from mlptools.common.structure import Structure
from mlptools.mlpgen.io import ReadFeatureParams

class TrainStructure:
    """Class to store training data structure
    """
    def __init__(self, fnames:str, with_force, weight):
        vasprun_array = [Vasprun(vasp_path) for ref_file in fnames \
                              for vasp_path in np.loadtxt(ref_file, dtype=str)[:, 1]]
        self.e_array = [v.get_energy() for v in vasprun_array]
        self.f_array = [np.ravel(v.get_forces(), order='F') for v in vasprun_array]
        struct_array = [tuple(v.get_structure()) for v in vasprun_array]
        self.vol_array = [st[3] for st in struct_array]
        self.s_array = [self.extract_s(v.get_stress() * vol / 1602.1766208) \
                        for v, vol in zip(vasprun_array, self.vol_array)]
        self.st_array = [Structure(st[0], st[1], st[2], st[4], types=st[5])\
                         for st in struct_array]

    def extract_s(self, s):
        """Extract xx, yy, zz, xy, yz, zx components from Stress Tensor.

        Args:
            s (multi_list): Stress Tensor

        Returns:
            list: xx, yy, zz, xy, yz, zx in order
        """
        return [s[0][0], s[1][1], s[2][2], s[0][1], s[1][2], s[2][0]]

    def correct_energy(self, atom_e):
        """Correct e_array by using the energy of isolated atoms.

        Args:
            atom_e (list): isolated atoms energy
        """
        self.e_array = [e - np.inner(st.n_atoms, atom_e) \
                        for e, st in zip(self.e_array, self.st_array)]

    def flat_array(self):
        """Flaten multi-list-type class properties
        """
        f_array = copy.deepcopy(self.f_array)
        s_array = copy.deepcopy(self.s_array)
        self.f_array = np.reshape(f_array, -1, order='C')
        self.s_array = np.reshape(s_array, -1, order='C')


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, required=True, \
                        help='Input file name. Training is performed from vasprun files.')
    args = parser.parse_args()

    p = InputParams(args.infile)
    di = ReadFeatureParams(p).get_params()

    start = time.time()
    tr = TrainStructure(di.train_names, di.train_force, di.train_weight)
    tr.flat_array()
    tr.correct_energy(di.atom_e)
    end = time.time()
    print(str(end-start)+"(s)")
