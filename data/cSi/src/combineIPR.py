#!/usr/bin/env python
import sys

def createdata(filename,output):

    with open(filename, "r") as f:
        lines = f.readlines()
    with open(output, "a") as w:
        newline=[]
        for line in lines:
            line=line.split()[1]
            newline.append(line)
        newline='\n'.join(newline)
        w.write(newline+'\n')

if __name__ == "__main__":
    output='../../../label.txt'
    for file in sys.argv[1:]:
        createdata(file,output)
