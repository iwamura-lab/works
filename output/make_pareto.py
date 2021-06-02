#!/usr/bin/env python

import os
from csv import reader

if __name__ == "__main__" :

    csv_path = os.getenv("HOME")+"/mlp-Fe/output/Fe/pareto_optimal.csv"
    a = []
    f = open(csv_path)
    csv_iter = reader(f)
    for i in csv_iter:
        a.append(i[0])
    for i in range(len(a)-1):
        print("{},{}".format(a[i],a[i+1]))
