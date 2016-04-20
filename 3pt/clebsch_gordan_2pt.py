#!/usr/bin/python

import numpy as np
import cmath

################################################################################
# setup of lookup tables for subduction coefficients. ##########################
################################################################################
# copy-pasted from Bastians rotation.nb file
# Subduction coefficients to irreps of octohedral group for all momenta

sqrt2 = np.sqrt(2.)
sqrt3 = np.sqrt(3.)
sqrt6 = np.sqrt(6.)

################################################################################
# T1
T1 = [[[np.asarray([0, 0, 0], dtype=int), 1, 0, 0]], \
      [[np.asarray([0, 0, 0], dtype=int), 0, 1, 0]], \
      [[np.asarray([0, 0, 0], dtype=int), 0, 0, 1]]]
T1 = np.asarray(T1)

################################################################################
# A1
A1 = [[np.asarray([ 0, 0, 1], dtype=int),  0,         0,         1j], \
      [np.asarray([ 0, 0,-1], dtype=int),  0,         0,        -1j], \
      [np.asarray([ 1, 0, 0], dtype=int),  1j,        0,         0], \
      [np.asarray([-1, 0, 0], dtype=int), -1j,        0,         0], \
      [np.asarray([ 0, 1, 0], dtype=int),  0,         1j,        0], \
      [np.asarray([ 0,-1, 0], dtype=int),  0,        -1j,        0], \
      [np.asarray([ 0, 1, 1], dtype=int),  0,         1j/sqrt2,  1j/sqrt2], \
      [np.asarray([ 0, 1,-1], dtype=int),  0,         1j/sqrt2, -1j/sqrt2], \
      [np.asarray([ 0,-1,-1], dtype=int),  0,        -1j/sqrt2, -1j/sqrt2], \
      [np.asarray([ 0,-1, 1], dtype=int),  0,        -1j/sqrt2,  1j/sqrt2], \
      [np.asarray([ 1, 0, 1], dtype=int),  1j/sqrt2,  0,         1j/sqrt2], \
      [np.asarray([ 1, 0,-1], dtype=int),  1j/sqrt2,  0,        -1j/sqrt2], \
      [np.asarray([-1, 0,-1], dtype=int), -1j/sqrt2,  0,        -1j/sqrt2], \
      [np.asarray([-1, 0, 1], dtype=int), -1j/sqrt2,  0,         1j/sqrt2], \
      [np.asarray([ 1, 1, 0], dtype=int),  1j/sqrt2,  1j/sqrt2,  0], \
      [np.asarray([-1, 1, 0], dtype=int), -1j/sqrt2,  1j/sqrt2,  0], \
      [np.asarray([-1,-1, 0], dtype=int), -1j/sqrt2, -1j/sqrt2,  0], \
      [np.asarray([ 1,-1, 0], dtype=int),  1j/sqrt2, -1j/sqrt2,  0], \
      [np.asarray([ 1, 1, 1], dtype=int),  1j/sqrt3,  1j/sqrt3,  1j/sqrt3], \
      [np.asarray([-1, 1, 1], dtype=int), -1j/sqrt3,  1j/sqrt3,  1j/sqrt3], \
      [np.asarray([-1,-1, 1], dtype=int), -1j/sqrt3, -1j/sqrt3,  1j/sqrt3], \
      [np.asarray([ 1,-1, 1], dtype=int),  1j/sqrt3, -1j/sqrt3,  1j/sqrt3], \
      [np.asarray([-1,-1,-1], dtype=int), -1j/sqrt3, -1j/sqrt3, -1j/sqrt3], \
      [np.asarray([ 1,-1,-1], dtype=int),  1j/sqrt3, -1j/sqrt3, -1j/sqrt3], \
      [np.asarray([ 1, 1,-1], dtype=int),  1j/sqrt3,  1j/sqrt3, -1j/sqrt3], \
      [np.asarray([-1, 1,-1], dtype=int), -1j/sqrt3,  1j/sqrt3, -1j/sqrt3], \
      [np.asarray([ 0, 0, 2], dtype=int),  0,         0,         1j], \
      [np.asarray([ 0, 0,-2], dtype=int),  0,         0,        -1j], \
      [np.asarray([ 2, 0, 0], dtype=int), -1j,        0,         0], \
      [np.asarray([-2, 0, 0], dtype=int),  1j,        0,         0], \
      [np.asarray([ 0, 2, 0], dtype=int),  0,        -1j,        0], \
      [np.asarray([ 0,-2, 0], dtype=int),  0,         1j,        0]]
A1 = np.asarray(A1)

################################################################################
# E2
E2_1 = [[np.asarray([ 0, 0, 1], dtype=int), -1j,        0,         0], \
        [np.asarray([ 0, 0,-1], dtype=int),  1j,        0,         0], \
        [np.asarray([ 1, 0, 0], dtype=int),  0,         0,         1j], \
        [np.asarray([-1, 0, 0], dtype=int),  0,         0,         1j], \
        [np.asarray([ 0, 1, 0], dtype=int),  0,         0,         1j], \
        [np.asarray([ 0,-1, 0], dtype=int),  0,         0,         1j], \
        [np.asarray([ 1, 1, 1], dtype=int), -1j/sqrt6, -1j/sqrt6,  2j/sqrt6], \
        [np.asarray([-1, 1, 1], dtype=int),  1j/sqrt6, -1j/sqrt6,  2j/sqrt6], \
        [np.asarray([-1,-1, 1], dtype=int),  1j/sqrt6,  1j/sqrt6,  2j/sqrt6], \
        [np.asarray([ 1,-1, 1], dtype=int), -1j/sqrt6,  1j/sqrt6,  2j/sqrt6], \
        [np.asarray([-1,-1,-1], dtype=int),  1j/sqrt6,  1j/sqrt6, -2j/sqrt6], \
        [np.asarray([ 1,-1,-1], dtype=int), -1j/sqrt6,  1j/sqrt6, -2j/sqrt6], \
        [np.asarray([ 1, 1,-1], dtype=int), -1j/sqrt6, -1j/sqrt6, -2j/sqrt6], \
        [np.asarray([-1, 1,-1], dtype=int),  1j/sqrt6, -1j/sqrt6, -2j/sqrt6], \
        [np.asarray([ 0, 0, 2], dtype=int), -1j,        0,         0], \
        [np.asarray([ 0, 0,-2], dtype=int),  1j,        0,         0], \
        [np.asarray([ 2, 0, 0], dtype=int),  0,         0,         1j], \
        [np.asarray([-2, 0, 0], dtype=int),  0,         0,         1j], \
        [np.asarray([ 0, 2, 0], dtype=int),  0,         0,         1j], \
        [np.asarray([ 0,-2, 0], dtype=int),  0,         0,         1j]]
E2_1 = np.asarray(E2_1)

E2_2 = [[np.asarray([ 0, 0, 1], dtype=int),  0,       -1,       0], \
        [np.asarray([ 0, 0,-1], dtype=int),  0,       -1,       0], \
        [np.asarray([ 1, 0, 0], dtype=int),  0,       -1,       0], \
        [np.asarray([-1, 0, 0], dtype=int),  0,        1,       0], \
        [np.asarray([ 0, 1, 0], dtype=int),  1,        0,       0], \
        [np.asarray([ 0,-1, 0], dtype=int), -1,        0,       0], \
        [np.asarray([ 1, 1, 1], dtype=int),  1/sqrt2, -1/sqrt2, 0], \
        [np.asarray([-1, 1, 1], dtype=int),  1/sqrt2,  1/sqrt2, 0], \
        [np.asarray([-1,-1, 1], dtype=int), -1/sqrt2,  1/sqrt2, 0], \
        [np.asarray([ 1,-1, 1], dtype=int), -1/sqrt2, -1/sqrt2, 0], \
        [np.asarray([-1,-1,-1], dtype=int),  1/sqrt2, -1/sqrt2, 0], \
        [np.asarray([ 1,-1,-1], dtype=int),  1/sqrt2,  1/sqrt2, 0], \
        [np.asarray([ 1, 1,-1], dtype=int), -1/sqrt2,  1/sqrt2, 0], \
        [np.asarray([-1, 1,-1], dtype=int), -1/sqrt2, -1/sqrt2, 0], \
        [np.asarray([ 0, 0, 2], dtype=int),  0,       -1,       0], \
        [np.asarray([ 0, 0,-2], dtype=int),  0,       -1,       0], \
        [np.asarray([ 2, 0, 0], dtype=int),  0,       -1,       0], \
        [np.asarray([-2, 0, 0], dtype=int),  0,        1,       0], \
        [np.asarray([ 0, 2, 0], dtype=int),  1,        0,       0], \
        [np.asarray([ 0,-2, 0], dtype=int), -1,        0,       0]]
E2_2 = np.asarray(E2_2)

#################################################################################
# B1
B1 = [[np.asarray([ 0, 1, 1], dtype=int), -1j,        0,        0], \
      [np.asarray([ 0, 1,-1], dtype=int),  1j,        0,        0], \
      [np.asarray([ 0,-1,-1], dtype=int), -1j,        0,        0], \
      [np.asarray([ 0,-1, 1], dtype=int),  1j,        0,        0], \
      [np.asarray([ 1, 0, 1], dtype=int),  0,        1j,        0], \
      [np.asarray([ 1, 0,-1], dtype=int),  0,        1j,        0], \
      [np.asarray([-1, 0,-1], dtype=int),  0,       -1j,        0], \
      [np.asarray([-1, 0, 1], dtype=int),  0,       -1j,        0], \
      [np.asarray([ 1, 1, 0], dtype=int),  0,        0,        1j], \
      [np.asarray([-1, 1, 0], dtype=int),  0,        0,       -1j], \
      [np.asarray([-1,-1, 0], dtype=int),  0,        0,       -1j], \
      [np.asarray([ 1,-1, 0], dtype=int),  0,        0,        1j]]
B1 = np.asarray(B1)

################################################################################
# B2
B2 = [[np.asarray([ 0, 1, 1], dtype=int),  0,        -1/sqrt2,  1/sqrt2], \
      [np.asarray([ 0, 1,-1], dtype=int),  0,        -1/sqrt2, -1/sqrt2], \
      [np.asarray([ 0,-1,-1], dtype=int),  0,         1/sqrt2, -1/sqrt2], \
      [np.asarray([ 0,-1, 1], dtype=int),  0,         1/sqrt2,  1/sqrt2], \
      [np.asarray([ 1, 0, 1], dtype=int), -1/sqrt2,   0,        1/sqrt2], \
      [np.asarray([ 1, 0,-1], dtype=int),  1/sqrt2,   0,        1/sqrt2], \
      [np.asarray([-1, 0,-1], dtype=int), -1/sqrt2,   0,        1/sqrt2], \
      [np.asarray([-1, 0, 1], dtype=int),  1/sqrt2,   0,        1/sqrt2], \
      [np.asarray([ 1, 1, 0], dtype=int),  1/sqrt2,  -1/sqrt2, 0], \
      [np.asarray([-1, 1, 0], dtype=int), -1/sqrt2,  -1/sqrt2, 0], \
      [np.asarray([-1,-1, 0], dtype=int),  1/sqrt2,  -1/sqrt2, 0], \
      [np.asarray([ 1,-1, 0], dtype=int), -1/sqrt2,  -1/sqrt2, 0]]

B2 = np.asarray(B2)

################################################################################
def coefficients(irrep):
  if irrep is 'T1':
    return T1
  elif irrep is 'A1':
    return A1
  elif irrep is 'E2_1':
    return E2_1
  elif irrep is 'E2_2':
    return E2_2
  elif irrep is 'B1':
    return B1
  elif irrep is 'B2':
    return B2


