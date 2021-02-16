#!/home/iwamura/mlp-Fe/venv/bin/python
"""
Program to execute the regression of mlp about paramagnetic FCC Fe
"""

# import modules to test
import time

# import standard modules
import argparse
#import tqdm
import numpy as np

# from mlptools import some modules
from mlptools.common.fileio import InputParams
from mlptools.common.readvasp import Vasprun
from mlptools.mlpgen.io import ReadFeatureParams

class TrainStructure:
    def __init__(self, fnames:str, with_force, weight):
        self.vasprun_array = [Vasprun(vasp_path) for ref_file in fnames \
                              for vasp_path in np.loadtxt(ref_file, dtype=str)[:, 1]]
        self.e_array = [v.get_energy() for v in self.vasprun_array]
        self.f_array = [np.ravel(v.get_forces(), order='F') for v in self.vasprun_array]
        self.axis_array = [v.get_structure()[0] for v in self.vasprun_array]
        self.pos_array = [v.get_structure()[1] for v in self.vasprun_array]
        self.atmn_array = [v.get_structure()[2] for v in self.vasprun_array]
        self.vol_array = [v.get_structure()[3] for v in self.vasprun_array]
        self.elm_array = [v.get_structure()[4] for v in self.vasprun_array]
        self.type_array = [v.get_structure()[5] for v in self.vasprun_array]

    def extract_s(self, s):
        """Extract xx, yy, zz, xy, yz, zx components from Stress Tensor.

        Args:
            s (multi_list): Stress Tensor

        Returns:
            list: xx, yy, zz, xy, yz, zx in order
        """
        return [s[0][0], s[1][1], s[2][2], s[0][1], s[1][2], s[2][0]]


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
