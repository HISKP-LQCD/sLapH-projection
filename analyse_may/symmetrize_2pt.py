#!/usr/bin/python

import math
import numpy as np

import sys
#sys.stdout = open('./irrep_and_gevp.out', 'w')

import utils

################################################################################
# some global variables ########################################################

momenta = [0,1,2,3,4] # momenta to analyse
T = 48

################################################################################
# reading data #################################################################

################################################################################
# 2pt charged pion for pion mass

## write all (anti-)symmetrized gevp eigenvalues
#path = './bootdata/p%1i/p%1i_sym_gevp' % (p, p)
#np.save(path, pc)
#path = './bootdata/p%1i/p%1i_sym_gevp_quantum_numbers' % (p, p)
## TODO: save t0 in qn

for p in momenta:
  filename = '../pion_mass/bootdata/p%d/C2+_p%d_real.npy' % (p, p)
  m_pi = np.load(filename)
  m_pi = np.mean(m_pi, axis=0)
  sym = np.zeros((T/2, m_pi.shape[-1]))
  sym[0] = m_pi[0]
  for t in range(1,T/2):
    sym[t] = (m_pi[t] + m_pi[T-t])/2

  sym = sym.T
  print sym.shape
  
  ################################################################################
  # Write data to disc ###########################################################
  
  ################################################################################
  utils.ensure_dir('./bootdata')
  utils.ensure_dir('./bootdata/p%1i' % p)
  
  ################################################################################
  path = './bootdata/p%1i/Mpi_p%1i_sym' % (p, p)
  np.save(path, sym)
  
