#!/usr/bin/env python
import sys

def createlabel(filename,output):
    with open(filename, "r") as f:
        lines = f.readlines()
    with open(output, "a") as w:
        newline=[]
        for i in range(2,len(lines)):
            line=lines[i].split()[0:4]
            if line[0]=="Si":
                line[0]='0'
            elif line[0]=="H":
                line[0]='1'
            newline+=line
        newline=' '.join(newline)
        w.write(newline+'\n')

if __name__ == "__main__":
    output='../../../src/data.txt'
    for file in sys.argv[1:]:
        createlabel(file,output)
