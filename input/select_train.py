#!/usr/bin/env python
"""
Program to make input file, train_vasprun_Fe. Use this program
after execution of select_test.py and with standard output redirect.
"""

# import standard modules
import pickle

if __name__ == "__main__" :
    fname_prefix = "/home/iwamura/mlp-Fe/3-dft/finished/"
    fname_list = pickle.load(open("shared/pickle.dump", "rb"))
    for fnum in fname_list :
        print(fname_prefix+str(fnum).zfill(5)+"/vasprun.xml")
