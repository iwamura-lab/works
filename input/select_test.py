#!/usr/bin/env python
"""
Program to make input file, test_vasprun_Fe.
Use this program with standard output redirection such as
    ./select_test.py > test_vasprun_Fe
"""

# import standard modules
import os
import random
import pickle

if __name__ == "__main__" :
    fname_prefix = os.getenv("HOME")+"/mlp-Fe/3-dft/finished/"
    fnum_list = [[(i+1) + 10*j for i in range(10)] for j in range(500)]
    train_fnum_list = [i+1 for i in range(5000)]
    # output the path of selected vasprun.xml for test as standard output
    for divided in fnum_list :
        fnum = random.choice(divided)
        print(fname_prefix+str(fnum).zfill(5)+"/vasprun.xml_to_mlip")
        train_fnum_list.remove(fnum)

    # serialize object
    pickle.dump(train_fnum_list, open("shared/pickle.dump", "wb"))
