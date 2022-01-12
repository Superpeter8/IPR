#!/usr/bin/env python

import sys

if __name__ == "__main__":
    output='output.txt'

    for file in sys.argv[1:]:
        with open(file, "r") as f:
            lines = f.readlines()
        with open(output, "a") as w:
            newline=[]
            for line in lines:
                line=line.split()[1]
                newline.append(line)
            newline='\n'.join(newline)
            w.write(newline+'\n')
