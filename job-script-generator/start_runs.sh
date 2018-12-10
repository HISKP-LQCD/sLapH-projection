#!/bin/bash

for momentum in 0 1 2 3 4; do
  for diagram in C20 C3c C4cB C4cD; do
    sbatch "read_qbig_slurm_p${momentum}_${diagram}.sh"
  done
done
