#!/home/iwamura/mlp-Fe/my_works/venv/bin/python3
"""
Program to count atoms in POSCAR files
"""

# import standard modules
import os
import numpy as np

# import modules made by Seko sensei
from mlptools.common.readvasp import Poscar


if __name__ == "__main__" :
    paths = os.listdir("/home/iwamura/mlp-Fe/3-dft/init/")
    atom_numbers = np.array([sum(Poscar(poscarfile).n_atoms) for poscarfile in paths])
    if np.any(atom_numbers != 32):
        print("There is at least one file without 32 atoms.")
