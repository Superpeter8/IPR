#!/usr/bin/env python

import sys
import numpy as np
import ase
from ase import Atoms
from ase.io import read, write

if __name__ == "__main__":
    with open('sample.in', "r") as fh:
        header = fh.readlines()
    for file in sys.argv[1:]:
        series=read(file,index=':',format='lammps-dump-text',order=True)
        for atoms in series:
            for atom in atoms:
                if atom.symbol=='H':
                    atom.symbol='Si'
            output='qein/'+file[11:-5]+'-'+str(series.index(atoms))+'.in'
            write(output,atoms,format='espresso-in')
            with open(output, "r") as f:
                lines = f.readlines()
            with open(output, "w+") as w:
                for i in range(len(header)):
                    if i == 2:
                        w.write("  prefix = '"+output[:-3].split('/')[-1]+"',\n")
                    else:
                        w.write(header[i])
                for i in range(len(lines)):
                    if i not in range(19):
                        w.write(lines[i])
