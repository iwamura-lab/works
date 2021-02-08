"""
Program to count atoms in POSCAR files
"""

# import standard modules
import argparse
import numpy as np

def extract_atoms(path):
    """Extract info of atom counts from POSCAR

    Args:
        path (str): path to POSCAR

    Returns:
        list: atom counts
    """
    f = open("/home/iwamura/mlp-Fe/3-dft/init/"+path)
    _ = [f.readline() for i in range(6)]
    atoms = [int(n) for n in f.readline().split()]
    f.close()
    return sum(atoms)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=str, help="File paths to count atoms. e.g.) 1-5000.")
    args = parser.parse_args()
    start, end = args.paths.split("-")
    atom_counts_dict = [extract_atoms("poscar-"+"{:0>5}".format(file_number))
                        for file_number in range(int(start), int(end)+1)]
    if np.any(atom_counts_dict != 32):
        print("There is at least one POSCAR without 32 atoms.")
    else :
        print("All POSCAR files have 32 atoms.")
