#!/usr/bin/python

import numpy as np
import cmath

import clebsch_gordan as cg
import utils as utils

# loop over irrep
# loop over row
# loop over qn
# if qn belongs to row: 
# add correlator to subduced

################################################################################
# read original data and call bootrap procedure

diagrams = ['C4+B', 'C4+D']
for diagram in diagrams:
  path = './readdata/p%1i/%s_p%1i.npy' % (p, diagram, p)
  read = np.load(path)
  path = './readdata/p%1i/%s_p%1i_quantum_numbers.npy' % (p, diagram, p)
  qn_data = np.load(path)
  if ( (qn_data.shape[0] != read.shape[0])):
    print '\tBootstrapped operators do not aggree with expected operators'
    exit(0)


