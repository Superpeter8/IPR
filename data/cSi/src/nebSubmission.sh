#!/bin/bash

export OMP_NUM_THREADS=2
T=1200
for R in 0.8 1; do 
mpirun -np 4 lmp -sf omp -var s 4 -var R $R -var T $T -var finaldump R$R-T$T.xyz -var d R$R-T$T-4.dump -in local_heating.in
done
