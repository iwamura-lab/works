#!/usr/bin/env python
"""
Program to combine and use the functions of Phonopy
"""

import argparse
from lammps_api.lammps_command import LammpsCommand

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="POSCAR", help="structure file name to read")
    args = parser.parse_args()

    lmp = LammpsCommand(structure_file="structure_equiv")
    lmp_st = lmp.get_structure()
    axis, positions, n_atoms, elements, types = lmp_st.get_structure()
    unitcell = PhonopyAtoms(elements, cell=axis.T, scaled_positions=positions.T)
