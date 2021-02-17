#!/usr/bin/env python
"""
Program to count atoms in POSCAR files
"""

# import standard modules
import os
import tqdm
import numpy as np

# import modules made by Seko sensei
from mlptools.common.readvasp import Poscar


if __name__ == "__main__" :
    poscar_dir = "/home/iwamura/mlp-Fe/3-dft/init/"
    paths = os.listdir(poscar_dir)
    atom_numbers = np.array([sum(Poscar(poscar_dir+poscarfile).n_atoms)\
                            for poscarfile in tqdm.tqdm(paths)])
    if np.any(atom_numbers != 32):
        print("There is at least one file without 32 atoms.")
    else:
        print("All the POSCAR files have 32 atoms.")
