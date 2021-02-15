#!/home/iwamura/mlp-Fe/venv/bin python
"""
Program to execute the regression of mlp about paramagnetic FCC Fe
"""

# import standard modules
import argparse

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, required=True, \
                        help='Input file name. Training is performed from vasprun files.')
