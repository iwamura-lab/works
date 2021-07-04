#!/usr/bin/env python

# import standard modules
import csv
import os
import argparse

from pymatgen.analysis.eos import Murnaghan

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, \
        default="MLP1/mlp1.csv", help="Input csv file for EOS fitting")
    args = parser.parse_args()

    csv_path = os.getenv("HOME")+"/mlp-Fe/pareto/"+args.file
    f = open(csv_path)
    volume = []
    energy = []
    for line in csv.reader(f) :
        volume.append(float(line[0])**3)
        energy.append(float(line[1]))
    f.close()
    murnaghan = Murnaghan(volume, energy)
    murnaghan.fit()
    print(murnaghan.v0**(1/3))
