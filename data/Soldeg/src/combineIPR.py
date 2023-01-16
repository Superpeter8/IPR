#!/usr/bin/env python
import sys

def createlabel(filename,output):

    with open(filename, "r") as f:
        lines = f.readlines()
    with open(output, "a") as w:
        newline=[]
        for i,line in enumerate(lines):
            line=line.split()[1]
            if line=='0' and i+1<len(lines):
                if lines[i+1].split()[1]=='0':
                    break
            newline.append(line)
        n=len(newline)
        newline='\n'.join(newline)
        w.write(newline+'\n')
    return n

if __name__ == "__main__":
    output='../../../label.txt'
    for file in sys.argv[1:]:
        createlabel(file,output)
