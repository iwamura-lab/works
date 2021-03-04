#!/usr/bin/env python
"""
Program to execute the regression of mlp about paramagnetic FCC Fe
"""

# import standard modules
import copy
import argparse
#import time
#import tqdm
import random
import numpy as np

# from mlptools import some modules
from mlptools.common.fileio import InputParams
from mlptools.mlpgen.regression import PotEstimation
from mlptools.common.readvasp import Vasprun
from mlptools.common.structure import Structure
from mlptools.mlpgen.io import ReadFeatureParams
from mlptools.mlpgen.model import Terms

def rearange_L(array, index_array):
    """Move designated columns to the head of array.

    Args:
        array (ndarray): input array
        index_array (list): list of index by which columns are designated

    Returns:
        ndarray: changed array
    """
    rest = np.delete(array, index_array, 1)
    return np.hstack((array[:, index_array], rest))

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
        self.with_force = with_force
        self.weight = weight

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
        """Flaten multi-list-type class properties.
        """
        f_array = copy.deepcopy(self.f_array)
        s_array = copy.deepcopy(self.s_array)
        self.f_array = np.reshape(f_array, -1, order='C')
        self.s_array = np.reshape(s_array, -1, order='C')

class VirtualDataInput:
    """Generate a new DataInput from normal DataInput
    """
    def __init__(self, di):
        self.vdi = copy.deepcopy(di)
        self.vdi.n_type = 2

    def get_data_input(self):
        """Return a newly generated DataInput.

        Returns:
            DataInput: virtual data structure
        """
        return self.vdi

class MagneticStructuralFeatures:
    """Data structure including magnetic structural features
    """
    def __init__(self, tr, spin_array, vdi):
        st_set_all_train = self.get_virtual_structures(tr.train, spin_array)
        n_st_dataset = [len(data.st_set) for data in tr.train]
        term = Terms(st_set_all_train, vdi, n_st_dataset, vdi.train_force)
        self.train_x_pm = term.get_x()

    def get_virtual_structures(self, dataset, spin_array):
        """Generate virtual structures from dataset based on spin_array and return the
           list of them

        Args:
            dataset (dr_array): array of the instances, DataRegression
            spin_array (ndarray): which spin each atom has

        Returns:
            list: list of rewrited structures
        """
        index_array = np.nonzero(spin_array==1)[0]
        n_atom_1 = len(index_array)
        n_atom_2 = sum(dataset[0].st_set[0].n_atoms) - len(index_array)
        n_atoms = [n_atom_1, n_atom_2]
        specie1 = ["A" for i in range(n_atom_1)]
        specie2 = ["B" for i in range(n_atom_2)]
        elements = specie1.extend(specie2)
        type1 = [0 for i in range(n_atom_1)]
        type2 = [1 for i in range(n_atom_2)]
        types = type1.extend(type2)
        st_list = [Structure(st.axis, rearange_L(st.positions, index_array), n_atoms, elements, \
                   types=types, comment=st.comment) for data in dataset for st in data.st_set]
        return st_list

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, required=True, \
        help='Input file name. Training is performed from vasprun files.')
    parser.add_argument('-p', '--pot', type=str, \
        default='mlp.pkl', help='Potential file name for mlptools')
    args = parser.parse_args()

    # prepare temporary spin array
    spin_array = np.array(random.choices([-1, 1], k=32))

    p = InputParams(args.infile)
    di = ReadFeatureParams(p).get_params()
    vdi = VirtualDataInput(di).get_data_input()

    # calculate structural features
    tr = PotEstimation(di=di)
    MagneticStructuralFeatures(tr, spin_array, vdi)
    tr.set_regression_data()
    # calculate magnetic structural features
