#!/usr/bin/env python

# import standard module
import shutil

if __name__ == "__main__" :
    f = open("./original/POSCAR", "r")
    content = [column for column in f]
    f.close()
    for i in range(99) :
        content[1] = str(7.00 + (i+1) / 10**4) + "\n"
        fpath = "7_00" + str(i+1).zfill(2)
        shutil.copytree("./original", "./"+fpath)
        f = open(fpath+"/POSCAR", "w")
        for column in content :
            print(column, file=f, end=" ")
        f.close()
