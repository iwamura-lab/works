#!/usr/bin/env python
"""
Program to make input file, train_vasprun_Fe.
Use this program after execution of select_test.py
and with standard output redirection such as
    ./select_train.py > train_vasprun_Fe.
"""

# import standard modules
import pickle

if __name__ == "__main__" :
    fname_prefix = "/home/iwamura/mlp-Fe/3-dft/finished/"
    fnum_list = pickle.load(open("shared/pickle.dump", "rb"))
    for fnum in fnum_list :
        print(fname_prefix+str(fnum).zfill(5)+"/vasprun.xml_to_mlip")
