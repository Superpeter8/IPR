#!/usr/bin/env python
import glob, os
from analyzebond import *
from combinexyz import *
from combineIPR import *

if __name__ == "__main__":
    data="data.txt"
    label="label.txt"
    if os.path.exists(data):
        os.remove(data)
    if os.path.exists(label):
        os.remove(label)
    with open(data, 'w') as fp:
        pass
    with open(label, 'w') as fp:
        pass
    prefices=[]
    n=0
    for file in glob.glob('../IPR/*.txt'):
        with open(file, "r") as f:
            lines = f.readlines()
        # if [line.split(' ')[1]=='0' for line in lines]:
        if lines[0].split()[1]=='0':
            # print(lines[0])
            # print(file)
            continue
        prefix=file.split('/')[-1][:-13]
        prefices.append(prefix)
        natoms=createneighborlist('../qein/'+prefix+'scf.in')
        nlabel=createlabel(file,label)
        try:
            assert(nlabel==natoms)
        except:
            print(prefix+' natoms='+str(natoms)+' nlabel='+str(nlabel))
        n=n+nlabel
    for prefix in prefices:
        for file in glob.glob('../neighborlist/'+prefix+'scf.in/*.xyz'):
            createdata(file,data)
    print(n)
