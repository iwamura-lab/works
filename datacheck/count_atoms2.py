"""
Program to count atoms in POSCAR files
"""

# import standard modules
import argparse

# import modules made by Seko sensei
from mlptools.common.readvasp import Poscar

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
    parser.add_argument("path", type=str, help="File path to count atoms. e.g.) poscar-03701.")
    args = parser.parse_args()
    st = Poscar(args.path).get_structure()
    atoms = sum(st[2])
    if atoms != 32 :
        print(args.path)
