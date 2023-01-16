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
    for file in glob.glob('../IPR/*/*.txt'):
        with open(file, "r") as f:
            lines = f.readlines()
        # if [line.split(' ')[1]=='0' for line in lines]:
        if lines[0].split()[1]=='0':
            # print(lines[0])
            # print(file)
            continue
        prefix='/'.join(file.split('/')[-2:])[:-13]
        prefices.append(prefix)
        createneighborlist('../qein/'+prefix+'.in')
        createlabel(file,label)
    for prefix in prefices:
        for file in glob.glob('../neighborlist/'+prefix+'.in/*.xyz'):
            createdata(file,data)
