#!/home/iwamura/mlp-Fe/venv/bin/python
"""
Program to execute the regression of mlp about paramagnetic FCC Fe
"""

# import modules to add functions
import tqdm
import time

# import standard modules
import argparse
import numpy as np

# from mlptools import some modules
from mlptools.common.fileio import InputParams
from mlptools.common.readvasp import Vasprun
from mlptools.mlpgen.io import ReadFeatureParams

class TrainStructure:
    def __init__(self, fnames:str, with_force, weight):
        self.vasprun_array = [Vasprun(vasp_path) for ref_file in fnames \
                              for vasp_path in tqdm.tqdm(np.loadtxt(ref_file, dtype=str)[:, 1])]

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, required=True, \
                        help='Input file name. Training is performed from vasprun files.')
    args = parser.parse_args()

    p = InputParams(args.infile)
    di = ReadFeatureParams(p).get_params()

    start = time.time()
    tr = TrainStructure(di.train_names, di.train_force, di.train_weight)
    end = time.time()
    print(str(end-start)+"(s)")
