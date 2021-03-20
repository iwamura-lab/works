#!/usr/bin/env python
"""
Program for the regression of mlp about paramagnetic FCC Fe
"""

# import standard modules
import copy
import argparse
#import time
#import tqdm
import numpy as np

# from mlptools import some modules
from mlptools.common.fileio import InputParams
from mlptools.mlpgen.regression import PotEstimation
from mlptools.common.structure import Structure
from mlptools.mlpgen.io import ReadFeatureParams, read_regression_params
from mlptools.mlpgen.model import Terms

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, required=True, \
        help='Input file name. Training is performed from vasprun files.')
    parser.add_argument('-d', '--read_data', type=str,\
        help='Training data file name. Training is performed from data.')
    parser.add_argument('--write_data', type=str, help='Saving training data.')
    parser.add_argument('--svd', action='store_true', \
        help='Use SVD to estimate ridge regression coefficients.')
    parser.add_argument('--noreg', action='store_true', \
        help='No regression mode.')
    parser.add_argument('-p', '--pot', type=str, \
        default='mlp.pkl', help='Potential file name for mlptools')
    parser.add_argument('-l', '--lammps', type=str,\
        default='mlp.lammps', help='Potential file name for lammps')
    args = parser.parse_args()

    p = InputParams(args.infile)
    di = ReadFeatureParams(p).get_params()

    # Make normal DataInput, which don't have different atoms
    # They have no information about vasprun.xml.
    normal_di = copy.deepcopy(di)
    normal_di.n_type = 1

    # calculate spin structural features
    tr = PotEstimation(di=di)

    # calculate normal structural features
    # calculation of train_x
    normal_types = [0 for i in range(32)]
    st_set_all_train = [Structure(st.axis, st.positions, [32], elements=st.elements, \
               types=normal_types, comment=st.comment) for data in tr.train for st in data.st_set]
    n_st_dataset_train = [len(data.st_set) for data in tr.train]
    term = Terms(st_set_all_train, normal_di, n_st_dataset_train, normal_di.train_force)
    train_x = np.hstack((tr.train_x, term.get_x()))
    tr.train_x = train_x

    # calculation of test_x
    st_set_all_test = [Structure(st.axis, st.positions, [32], elements=st.elements, \
               types=normal_types, comment=st.comment) for data in tr.test for st in data.st_set]
    n_st_dataset_test = [len(data.st_set) for data in tr.test]
    force_dataset = [di.wforce for v in di.test_names]
    term = Terms(st_set_all_test, normal_di, n_st_dataset_test, force_dataset)
    test_x = np.hstack((tr.test_x, term.get_x()))
    tr.test_x = test_x

    tr.set_regression_data()

    # start regression
    if args.noreg is False:
        reg_method, alpha_min, alpha_max, n_alpha = read_regression_params(p)
        if (reg_method == 'ridge' or reg_method == 'lasso'):
            pot = tr.regularization_reg(method=reg_method,alpha_min=alpha_min,\
                alpha_max=alpha_max,n_alpha=n_alpha,svd=args.svd)
        elif reg_method == 'normal':
            pot = tr.normal_reg()

        pot.save_pot(file_name=args.pot)
        pot.save_pot_for_lammps(file_name=args.lammps)

        print(' --- input parameters ----')
        pot.di.model_e.print()
        print(' --- best model ----')
        if (reg_method == 'ridge' or reg_method == 'lasso'):
            print(' alpha = ', tr.best_alpha)

        rmse_train_e, rmse_test_e, rmse_train_f, files_train, \
            rmse_test_f, rmse_train_s, rmse_test_s, files_test \
            = tr.get_best_rmse()

        print(' -- Prediction Error --')
        for re, rf, rs, f in \
            zip(rmse_train_e, rmse_train_f, rmse_train_s, files_train):
            print(' structures :', f)
            print(' rmse (energy, train) = ', re * 1000, ' (meV/atom)')
            if rf is not None:
                print(' rmse (force, train) = ', rf, ' (eV/ang)')
                print(' rmse (stress, train) = ', rs, ' (GPa)')
        for re, rf, rs, f in \
            zip(rmse_test_e, rmse_test_f, rmse_test_s, files_test):
            print(' structures :', f)
            print(' rmse (energy, test) = ', re * 1000, ' (meV/atom)')
            if rf is not None:
                print(' rmse (force, test) = ', rf, ' (eV/ang)')
                print(' rmse (stress, test) = ', rs, ' (GPa)')
