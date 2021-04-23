#!/usr/bin/env python
"""
Program to combine and use the functions of Phonopy
"""

# import standard modules
import argparse
#import numpy as np
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
#from pymatgen.io.vasp import Vasprun

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vasprun", default="vasprun.xml", \
                        help="The path of vasprun.xml, used to get forces on atoms")
    args = parser.parse_args()

    phonon = phonopy.load("phonopy_params.yaml")
    qpath = [[[0, 0, 0], [0.5, 0, 0.5]], [[0.5, 0, 0.5], [0.375, 0.375, 0.75], [0, 0, 0]], \
             [[0, 0, 0], [0.5, 0.5, 0.5]]]
    labels = ["$\\Gamma$", "X", "X", "K", "$\\Gamma$", "$\\Gamma$", "L"]
    qpoints, connections = get_band_qpoints_and_path_connections(qpath)
    phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
    phonon.plot_band_structure().show()

    #vasp = Vasprun(args.vasprun)
    #forces = vasp.get_trajectory().site_properties[0]["forces"]
    #displacement = np.array([0.01, 0, 0])
    #supercell_dataset = {"number": 0, "displacement": displacement, "forces": np.array(forces)}
    #displacement_dataset = {"natom": sum(n_atoms), "first_atoms": [supercell_dataset]}
    #phonon.dataset = displacement_dataset
    #phonon.produce_force_constants()
    #phonon.save()
    #phonon.auto_band_structure(write_yaml=True)
