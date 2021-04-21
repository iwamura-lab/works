#!/usr/bin/env python
"""
Modules about operation of structure data or structure file
"""

# import standard modules
import sys
import numpy as np

def division_and_sliding(a, b):
    """Calculation of 'dividend / divisor' and adjustment of the value into [0, 1]

    Args:
        a (float): dividend
        b (float): divisor

    Returns:
        float: result value
    """
    result = a / b
    if result < 0:
        result += 1
    elif result > 1:
        result -= 1
    return result

class LammpsStructure:
    """Class to treat lammps structure file(only Fm3m)
    """

    def __init__(self, filename="structure_equiv", element="Fe"):

        self.element = element

        f = open(filename)
        lines = f.readlines()
        f.close()

        self.n_atoms = int(lines[2].split()[0])
        self.lt_const = lt_const = float(lines[5].split()[1])
        self.axis = np.eye(3) * lt_const
        self.positions = np.array([[i for i in map(float, line.split()[2:])] \
                                    for line in lines[13:]])
        self.get_scaled_positions()

    def get_scaled_positions(self):
        """Generate fractional coordinates from cartesian coordinates
        """
        self.scaled_positions = np.array([[division_and_sliding(comp, self.lt_const) \
                                    for comp in position] for position in self.positions])

    def output_poscar(self, filename=None, comment="../1-ideal/perfect-sqs-32/POSCAR"):
        """Output POSCAR format file

        Args:
            filename (str, optional): Filename to output. Defaults to None.
            comment (str, optional): The comment of POSCAR's first line.
                                     Defaults to "../1-ideal/perfect-sqs-32/POSCAR".
        """
        if filename is None:
            f = sys.stdout
        else:
            f = open(filename, "w")

        print(comment, file=f)
        print("1.0", file=f)
        for v in self.axis:
            print("   {0:.15f} {1:.15f} {2:.15f}".format(v[0], v[1], v[2]), file=f)
        print(self.element, file=f)
        print(self.n_atoms, file=f)
        print("Direct", file=f)
        for p in self.scaled_positions:
            print("   {0:.15f} {1:.15f} {2:.15f}".format(p[0], p[1], p[2]), file=f)

if __name__ == "__main__":
    lmp_st = LammpsStructure()
    lmp_st.output_poscar()
