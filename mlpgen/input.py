#!/usr/bin/env python
"""
Program to get data structures used as input
"""

# from mlptools import some modules
from mlptools.common.readvasp import Vasprun
from mlptools.common.structure import Structure

class TrainStructure:
    """Class to store training data
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
