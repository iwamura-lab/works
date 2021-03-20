#!/usr/bin/env python
"""
Program to make input file, test_vasprun_Fe. Use this program with standard output redirect.
"""

# import standard modules
import random
import pickle

if __name__ == "__main__" :
    fname_prefix = "/home/iwamura/mlp-Fe/3-dft/finished/"
    fname_list = [[(i+1) + 5*j for i in range(5)] for j in range(1000)]
    train_fname_list = [i+1 for i in range(5000)]
    # standard output the path of vasprun.xml
    for i in fname_list :
        fnum = random.choice(i)
        print(fname_prefix+str(fnum).zfill(5)+"/vasprun.xml")
        train_fname_list.remove(fnum)

    # serialize object
    pickle.dump(train_fname_list, open("shared/pickle.dump", "wb"))
