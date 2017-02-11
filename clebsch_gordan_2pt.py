#!/usr/bin/python

import numpy as np
import cmath

import pandas as pd
from pandas import Series, DataFrame

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
#T1 = [[[(np.asarray([0, 0, 0], dtype=int),),  1j/sqrt2, -1./sqrt2, 0]], \
#      [[(np.asarray([0, 0, 0], dtype=int),),  0,         0,        1j]], \
#      [[(np.asarray([0, 0, 0], dtype=int),), -1j/sqrt2, -1./sqrt2, 0]]]
#T1 = np.asarray(T1)

# TODO: Do I need p?
T1 = DataFrame({'p' : [(0,0,0)]*9, \
                '|J, M\rangle' : 
                            ["|1,+1\rangle", "|1, 0\rangle", "|1,-1\rangle"]*3, \
                'cg-coefficient' : [ 1, 0, 0, \
                                     0, 1, 0, \
                                     0, 0, 1 ]}, \
                index=pd.Index([1]*3+[2]*3+[3]*3, name='\mu'))

#T1 = DataFrame({'p' : [(0,0,0)]*9, \
#                '\gamma' : [(1,), (2,), (3,)]*3, \
#                'cg-coefficient' : [ 1j/sqrt2, -1./sqrt2, 0, \
#                                     0,         0,        1j, \
#                                    -1j/sqrt2, -1./sqrt2, 0]}, \
#                index=pd.Index([1]*3+[2]*3+[3]*3, name='\mu'))

#T1 = DataFrame({'p' : [np.array([0,0,0])]*9, \
#                '\gamma' : [(1,), (2,), (3,)]*3, 
#                'cg-coefficient' : [ 1j/sqrt2, -1./sqrt2, 0, \
#                                     0,         0,        1j, \
#                                    -1j/sqrt2, -1./sqrt2, 0]}, \
#                index=pd.Index([1]*3+[2]*3+[3]*3, name='\row'))

# In the CM-Frame this is equivalent to only taking the diagonal combinations
#T1 = [[[np.asarray([0, 0, 0], dtype=int), 1, 0, 0]], \
#      [[np.asarray([0, 0, 0], dtype=int), 0, 1, 0]], \
#      [[np.asarray([0, 0, 0], dtype=int), 0, 0, 1]]]
#T1 = np.asarray(T1)

def coefficients(p_cm, irrep):

  ##############################################################################
  # T1
  T1 = DataFrame({'p' : [(0,0,0)]*9, \
                  '|J, M\rangle' : 
                           ["|1,+1\rangle", "|1, 0\rangle", "|1,-1\rangle"]*3, \
                  'cg-coefficient' : [ 1, 0, 0, \
                                       0, 1, 0, \
                                       0, 0, 1 ]}, \
                  index=pd.Index([1]*3+[2]*3+[3]*3, name='\mu'))
  
  ################################################################################
  # A1
  A1 = {}

  A1[1] = DataFrame([[( 0, 0, 1),  0,         0,         1j], \
         [( 0, 0,-1),  0,         0,        -1j], \
         [( 1, 0, 0),  1j,        0,         0], \
         [(-1, 0, 0), -1j,        0,         0], \
         [( 0, 1, 0),  0,         1j,        0], \
         [( 0,-1, 0),  0,        -1j,        0]])
         
  A1[2] = DataFrame([[( 0, 1, 1),  0,         1j/sqrt2,  1j/sqrt2], \
         [( 0, 1,-1),  0,         1j/sqrt2, -1j/sqrt2], \
         [( 0,-1,-1),  0,        -1j/sqrt2, -1j/sqrt2], \
         [( 0,-1, 1),  0,        -1j/sqrt2,  1j/sqrt2], \
         [( 1, 0, 1),  1j/sqrt2,  0,         1j/sqrt2], \
         [( 1, 0,-1),  1j/sqrt2,  0,        -1j/sqrt2], \
         [(-1, 0,-1), -1j/sqrt2,  0,        -1j/sqrt2], \
         [(-1, 0, 1), -1j/sqrt2,  0,         1j/sqrt2], \
         [( 1, 1, 0),  1j/sqrt2,  1j/sqrt2,  0], \
         [(-1, 1, 0), -1j/sqrt2,  1j/sqrt2,  0], \
         [(-1,-1, 0), -1j/sqrt2, -1j/sqrt2,  0], \
         [( 1,-1, 0),  1j/sqrt2, -1j/sqrt2,  0]])
  
  A1[3] = DataFrame([[( 1, 1, 1),  1j/sqrt3,  1j/sqrt3,  1j/sqrt3], \
         [(-1, 1, 1), -1j/sqrt3,  1j/sqrt3,  1j/sqrt3], \
         [(-1,-1, 1), -1j/sqrt3, -1j/sqrt3,  1j/sqrt3], \
         [( 1,-1, 1),  1j/sqrt3, -1j/sqrt3,  1j/sqrt3], \
         [(-1,-1,-1), -1j/sqrt3, -1j/sqrt3, -1j/sqrt3], \
         [( 1,-1,-1),  1j/sqrt3, -1j/sqrt3, -1j/sqrt3], \
         [( 1, 1,-1),  1j/sqrt3,  1j/sqrt3, -1j/sqrt3], \
         [(-1, 1,-1), -1j/sqrt3,  1j/sqrt3, -1j/sqrt3]])
  
  A1[4] = DataFrame([[( 0, 0, 2),  0,         0,         1j], \
         [( 0, 0,-2),  0,         0,        -1j], \
         [( 2, 0, 0), -1j,        0,         0], \
         [(-2, 0, 0),  1j,        0,         0], \
         [( 0, 2, 0),  0,        -1j,        0], \
         [( 0,-2, 0),  0,         1j,        0]])

  for p in range(1,5):
    A1[p] = pd.concat([DataFrame(A1[p][[0,g]].values, columns=['p', 'cg-coefficient']) for g in range(1,4)])
    A1[p].index = pd.Index([1] * len(A1[p]), name='\mu')
    A1[p]['|J, M\rangle'] = ["|1,+1\rangle"]*(len(A1[p])/3) + ["|1, 0\rangle"]*(len(A1[p])/3) + ["|1,-1\rangle"]*(len(A1[p])/3)
    A1[p].sort_index(inplace=True)

  ################################################################################
  # E2
  E2 = {}

  E2[1] = DataFrame([[( 0, 0, 1),  0,       -1,       0], \
         [( 0, 0,-1),  0,       -1,       0], \
         [( 1, 0, 0),  0,       -1,       0], \
         [(-1, 0, 0),  0,        1,       0], \
         [( 0, 1, 0),  1,        0,       0], \
         [( 0,-1, 0), -1,        0,       0], \

         [( 0, 0, 1), -1j,        0,         0], \
         [( 0, 0,-1),  1j,        0,         0], \
         [( 1, 0, 0),  0,         0,         1j], \
         [(-1, 0, 0),  0,         0,         1j], \
         [( 0, 1, 0),  0,         0,         1j], \
         [( 0,-1, 0),  0,         0,         1j]], index=pd.Index([1]*6+[2]*6, name='\mu'))

  E2[3] = DataFrame([[( 1, 1, 1),  1/sqrt2, -1/sqrt2, 0], \
         [(-1, 1, 1),  1/sqrt2,  1/sqrt2, 0], \
         [(-1,-1, 1), -1/sqrt2,  1/sqrt2, 0], \
         [( 1,-1, 1), -1/sqrt2, -1/sqrt2, 0], \
         [(-1,-1,-1),  1/sqrt2, -1/sqrt2, 0], \
         [( 1,-1,-1),  1/sqrt2,  1/sqrt2, 0], \
         [( 1, 1,-1), -1/sqrt2,  1/sqrt2, 0], \
         [(-1, 1,-1), -1/sqrt2, -1/sqrt2, 0], \

         [( 1, 1, 1), -1j/sqrt6, -1j/sqrt6,  2j/sqrt6], \
         [(-1, 1, 1),  1j/sqrt6, -1j/sqrt6,  2j/sqrt6], \
         [(-1,-1, 1),  1j/sqrt6,  1j/sqrt6,  2j/sqrt6], \
         [( 1,-1, 1), -1j/sqrt6,  1j/sqrt6,  2j/sqrt6], \
         [(-1,-1,-1),  1j/sqrt6,  1j/sqrt6, -2j/sqrt6], \
         [( 1,-1,-1), -1j/sqrt6,  1j/sqrt6, -2j/sqrt6], \
         [( 1, 1,-1), -1j/sqrt6, -1j/sqrt6, -2j/sqrt6], \
         [(-1, 1,-1),  1j/sqrt6, -1j/sqrt6, -2j/sqrt6]], index=pd.Index([1]*8+[2]*8, name='\mu'))

  E2[4] = DataFrame([[( 0, 0, 2),  0,       -1,       0], \
         [( 0, 0,-2),  0,       -1,       0], \
         [( 2, 0, 0),  0,       -1,       0], \
         [(-2, 0, 0),  0,        1,       0], \
         [( 0, 2, 0),  1,        0,       0], \
         [( 0,-2, 0), -1,        0,       0], \
                                            \
         [( 0, 0, 2), -1j,        0,         0], \
         [( 0, 0,-2),  1j,        0,         0], \
         [( 2, 0, 0),  0,         0,         1j], \
         [(-2, 0, 0),  0,         0,         1j], \
         [( 0, 2, 0),  0,         0,         1j], \
         [( 0,-2, 0),  0,         0,         1j]], index=pd.Index([1]*6 + [2]*6, name='\mu'))

  for p in [1,3,4]:
    E2[p] = pd.concat([DataFrame(E2[p][[0,g]].reset_index().values, columns=['\mu', 'p', 'cg-coefficient']) for g in range(1,4)]).set_index('\mu')
    E2[p]['|J, M\rangle'] = ["|1,+1\rangle"]*(len(E2[p])/3) + ["|1, 0\rangle"]*(len(E2[p])/3) + ["|1,-1\rangle"]*(len(E2[p])/3)
    E2[p].sort_index(inplace=True)
 
  ################################################################################
  # B1
  B1 = {}

  B1[2] = DataFrame([[( 0, 1, 1),  0,        -1/sqrt2,  1/sqrt2], \
         [( 0, 1,-1),  0,        -1/sqrt2, -1/sqrt2], \
         [( 0,-1,-1),  0,         1/sqrt2, -1/sqrt2], \
         [( 0,-1, 1),  0,         1/sqrt2,  1/sqrt2], \
         [( 1, 0, 1), -1/sqrt2,   0,        1/sqrt2], \
         [( 1, 0,-1),  1/sqrt2,   0,        1/sqrt2], \
         [(-1, 0,-1), -1/sqrt2,   0,        1/sqrt2], \
         [(-1, 0, 1),  1/sqrt2,   0,        1/sqrt2], \
         [( 1, 1, 0),  1/sqrt2,  -1/sqrt2, 0], \
         [(-1, 1, 0), -1/sqrt2,  -1/sqrt2, 0], \
         [(-1,-1, 0),  1/sqrt2,  -1/sqrt2, 0], \
         [( 1,-1, 0), -1/sqrt2,  -1/sqrt2, 0]])

  for p in [2]:
    B1[p] = pd.concat([DataFrame(B1[p][[0,g]].values, columns=['p', 'cg-coefficient']) for g in range(1,4)])
    B1[p].index = pd.Index([1] * len(B1[p]), name='\mu')
    B1[p]['|J, M\rangle'] = ["|1,+1\rangle"]*(len(B1[p])/3) + ["|1, 0\rangle"]*(len(B1[p])/3) + ["|1,-1\rangle"]*(len(B1[p])/3)
    B1[p].sort_index(inplace=True)
  
  #################################################################################
  # B2
  B2 = {}

  B2[2] = DataFrame([[( 0, 1, 1), -1j,        0,        0], \
         [( 0, 1,-1),  1j,        0,        0], \
         [( 0,-1,-1), -1j,        0,        0], \
         [( 0,-1, 1),  1j,        0,        0], \
         [( 1, 0, 1),  0,        1j,        0], \
         [( 1, 0,-1),  0,        1j,        0], \
         [(-1, 0,-1),  0,       -1j,        0], \
         [(-1, 0, 1),  0,       -1j,        0], \
         [( 1, 1, 0),  0,        0,        1j], \
         [(-1, 1, 0),  0,        0,       -1j], \
         [(-1,-1, 0),  0,        0,       -1j], \
         [( 1,-1, 0),  0,        0,        1j]])

  for p in [2]:
    B2[p] = pd.concat([DataFrame(B2[p][[0,g]].values, columns=['p', 'cg-coefficient']) for g in range(1,4)])
    B2[p].index = pd.Index([1] * len(B2[p]), name='\mu')
    B2[p]['|J, M\rangle'] = ["|1,+1\rangle"]*(len(B2[p])/3) + ["|1, 0\rangle"]*(len(B2[p])/3) + ["|1,-1\rangle"]*(len(B2[p])/3)
    B2[p].sort_index(inplace=True)
 
  if irrep is 'T1':
    return T1
  elif irrep is 'A1':
    return A1[p_cm]
  elif irrep is 'E2':
    return E2[p_cm]
  elif irrep is 'B1':
    return B1[p_cm]
  elif irrep is 'B2':
    return B2[p_cm]

################################################################################
#################################################################################
#################################################################################
## A1
#A1 = [[[(np.asarray([ 0, 0, 1], dtype=int),),  0,         0,         1j], \
#       [(np.asarray([ 0, 0,-1], dtype=int),),  0,         0,        -1j], \ #       [(np.asarray([ 1, 0, 0], dtype=int),),  1j,        0,         0], \
#       [(np.asarray([-1, 0, 0], dtype=int),), -1j,        0,         0], \
#       [(np.asarray([ 0, 1, 0], dtype=int),),  0,         1j,        0], \
#       [(np.asarray([ 0,-1, 0], dtype=int),),  0,        -1j,        0], \
#       [(np.asarray([ 0, 1, 1], dtype=int),),  0,         1j/sqrt2,  1j/sqrt2], \
#       [(np.asarray([ 0, 1,-1], dtype=int),),  0,         1j/sqrt2, -1j/sqrt2], \
#       [(np.asarray([ 0,-1,-1], dtype=int),),  0,        -1j/sqrt2, -1j/sqrt2], \
#       [(np.asarray([ 0,-1, 1], dtype=int),),  0,        -1j/sqrt2,  1j/sqrt2], \
#       [(np.asarray([ 1, 0, 1], dtype=int),),  1j/sqrt2,  0,         1j/sqrt2], \
#       [(np.asarray([ 1, 0,-1], dtype=int),),  1j/sqrt2,  0,        -1j/sqrt2], \
#       [(np.asarray([-1, 0,-1], dtype=int),), -1j/sqrt2,  0,        -1j/sqrt2], \
#       [(np.asarray([-1, 0, 1], dtype=int),), -1j/sqrt2,  0,         1j/sqrt2], \
#       [(np.asarray([ 1, 1, 0], dtype=int),),  1j/sqrt2,  1j/sqrt2,  0], \
#       [(np.asarray([-1, 1, 0], dtype=int),), -1j/sqrt2,  1j/sqrt2,  0], \
#       [(np.asarray([-1,-1, 0], dtype=int),), -1j/sqrt2, -1j/sqrt2,  0], \
#       [(np.asarray([ 1,-1, 0], dtype=int),),  1j/sqrt2, -1j/sqrt2,  0], \
#       [(np.asarray([ 1, 1, 1], dtype=int),),  1j/sqrt3,  1j/sqrt3,  1j/sqrt3], \
#       [(np.asarray([-1, 1, 1], dtype=int),), -1j/sqrt3,  1j/sqrt3,  1j/sqrt3], \
#       [(np.asarray([-1,-1, 1], dtype=int),), -1j/sqrt3, -1j/sqrt3,  1j/sqrt3], \
#       [(np.asarray([ 1,-1, 1], dtype=int),),  1j/sqrt3, -1j/sqrt3,  1j/sqrt3], \
#       [(np.asarray([-1,-1,-1], dtype=int),), -1j/sqrt3, -1j/sqrt3, -1j/sqrt3], \
#       [(np.asarray([ 1,-1,-1], dtype=int),),  1j/sqrt3, -1j/sqrt3, -1j/sqrt3], \
#       [(np.asarray([ 1, 1,-1], dtype=int),),  1j/sqrt3,  1j/sqrt3, -1j/sqrt3], \
#       [(np.asarray([-1, 1,-1], dtype=int),), -1j/sqrt3,  1j/sqrt3, -1j/sqrt3], \
#       [(np.asarray([ 0, 0, 2], dtype=int),),  0,         0,         1j], \
#       [(np.asarray([ 0, 0,-2], dtype=int),),  0,         0,        -1j], \
#       [(np.asarray([ 2, 0, 0], dtype=int),), -1j,        0,         0], \
#       [(np.asarray([-2, 0, 0], dtype=int),),  1j,        0,         0], \
#       [(np.asarray([ 0, 2, 0], dtype=int),),  0,        -1j,        0], \
#       [(np.asarray([ 0,-2, 0], dtype=int),),  0,         1j,        0]]]
#A1 = np.asarray(A1)
#
#################################################################################
## E2
#E2 = [[[(np.asarray([ 0, 0, 1], dtype=int),),  0,       -1,       0], \
#       [(np.asarray([ 0, 0,-1], dtype=int),),  0,       -1,       0], \
#       [(np.asarray([ 1, 0, 0], dtype=int),),  0,       -1,       0], \
#       [(np.asarray([-1, 0, 0], dtype=int),),  0,        1,       0], \
#       [(np.asarray([ 0, 1, 0], dtype=int),),  1,        0,       0], \
#       [(np.asarray([ 0,-1, 0], dtype=int),), -1,        0,       0], \
#       [(np.asarray([ 1, 1, 1], dtype=int),),  1/sqrt2, -1/sqrt2, 0], \
#       [(np.asarray([-1, 1, 1], dtype=int),),  1/sqrt2,  1/sqrt2, 0], \
#       [(np.asarray([-1,-1, 1], dtype=int),), -1/sqrt2,  1/sqrt2, 0], \
#       [(np.asarray([ 1,-1, 1], dtype=int),), -1/sqrt2, -1/sqrt2, 0], \
#       [(np.asarray([-1,-1,-1], dtype=int),),  1/sqrt2, -1/sqrt2, 0], \
#       [(np.asarray([ 1,-1,-1], dtype=int),),  1/sqrt2,  1/sqrt2, 0], \
#       [(np.asarray([ 1, 1,-1], dtype=int),), -1/sqrt2,  1/sqrt2, 0], \
#       [(np.asarray([-1, 1,-1], dtype=int),), -1/sqrt2, -1/sqrt2, 0], \
#       [(np.asarray([ 0, 0, 2], dtype=int),),  0,       -1,       0], \
#       [(np.asarray([ 0, 0,-2], dtype=int),),  0,       -1,       0], \
#       [(np.asarray([ 2, 0, 0], dtype=int),),  0,       -1,       0], \
#       [(np.asarray([-2, 0, 0], dtype=int),),  0,        1,       0], \
#       [(np.asarray([ 0, 2, 0], dtype=int),),  1,        0,       0], \
#       [(np.asarray([ 0,-2, 0], dtype=int),), -1,        0,       0]], \
#                                                                    \
#      [[(np.asarray([ 0, 0, 1], dtype=int),), -1j,        0,         0], \
#       [(np.asarray([ 0, 0,-1], dtype=int),),  1j,        0,         0], \
#       [(np.asarray([ 1, 0, 0], dtype=int),),  0,         0,         1j], \
#       [(np.asarray([-1, 0, 0], dtype=int),),  0,         0,         1j], \
#       [(np.asarray([ 0, 1, 0], dtype=int),),  0,         0,         1j], \
#       [(np.asarray([ 0,-1, 0], dtype=int),),  0,         0,         1j], \
#       [(np.asarray([ 1, 1, 1], dtype=int),), -1j/sqrt6, -1j/sqrt6,  2j/sqrt6], \
#       [(np.asarray([-1, 1, 1], dtype=int),),  1j/sqrt6, -1j/sqrt6,  2j/sqrt6], \
#       [(np.asarray([-1,-1, 1], dtype=int),),  1j/sqrt6,  1j/sqrt6,  2j/sqrt6], \
#       [(np.asarray([ 1,-1, 1], dtype=int),), -1j/sqrt6,  1j/sqrt6,  2j/sqrt6], \
#       [(np.asarray([-1,-1,-1], dtype=int),),  1j/sqrt6,  1j/sqrt6, -2j/sqrt6], \
#       [(np.asarray([ 1,-1,-1], dtype=int),), -1j/sqrt6,  1j/sqrt6, -2j/sqrt6], \
#       [(np.asarray([ 1, 1,-1], dtype=int),), -1j/sqrt6, -1j/sqrt6, -2j/sqrt6], \
#       [(np.asarray([-1, 1,-1], dtype=int),),  1j/sqrt6, -1j/sqrt6, -2j/sqrt6], \
#       [(np.asarray([ 0, 0, 2], dtype=int),), -1j,        0,         0], \
#       [(np.asarray([ 0, 0,-2], dtype=int),),  1j,        0,         0], \
#       [(np.asarray([ 2, 0, 0], dtype=int),),  0,         0,         1j], \
#       [(np.asarray([-2, 0, 0], dtype=int),),  0,         0,         1j], \
#       [(np.asarray([ 0, 2, 0], dtype=int),),  0,         0,         1j], \
#       [(np.asarray([ 0,-2, 0], dtype=int),),  0,         0,         1j]]]
#E2 = np.asarray(E2)
#
#################################################################################
## B1
#B1 = [[[(np.asarray([ 0, 1, 1], dtype=int),),  0,        -1/sqrt2,  1/sqrt2], \
#       [(np.asarray([ 0, 1,-1], dtype=int),),  0,        -1/sqrt2, -1/sqrt2], \
#       [(np.asarray([ 0,-1,-1], dtype=int),),  0,         1/sqrt2, -1/sqrt2], \
#       [(np.asarray([ 0,-1, 1], dtype=int),),  0,         1/sqrt2,  1/sqrt2], \
#       [(np.asarray([ 1, 0, 1], dtype=int),), -1/sqrt2,   0,        1/sqrt2], \
#       [(np.asarray([ 1, 0,-1], dtype=int),),  1/sqrt2,   0,        1/sqrt2], \
#       [(np.asarray([-1, 0,-1], dtype=int),), -1/sqrt2,   0,        1/sqrt2], \
#       [(np.asarray([-1, 0, 1], dtype=int),),  1/sqrt2,   0,        1/sqrt2], \
#       [(np.asarray([ 1, 1, 0], dtype=int),),  1/sqrt2,  -1/sqrt2, 0], \
#       [(np.asarray([-1, 1, 0], dtype=int),), -1/sqrt2,  -1/sqrt2, 0], \
#       [(np.asarray([-1,-1, 0], dtype=int),),  1/sqrt2,  -1/sqrt2, 0], \
#       [(np.asarray([ 1,-1, 0], dtype=int),), -1/sqrt2,  -1/sqrt2, 0]]]
#
#B1 = np.asarray(B1)
#
##################################################################################
## B2
#B2 = [[[(np.asarray([ 0, 1, 1], dtype=int),), -1j,        0,        0], \
#       [(np.asarray([ 0, 1,-1], dtype=int),),  1j,        0,        0], \
#       [(np.asarray([ 0,-1,-1], dtype=int),), -1j,        0,        0], \
#       [(np.asarray([ 0,-1, 1], dtype=int),),  1j,        0,        0], \
#       [(np.asarray([ 1, 0, 1], dtype=int),),  0,        1j,        0], \
#       [(np.asarray([ 1, 0,-1], dtype=int),),  0,        1j,        0], \
#       [(np.asarray([-1, 0,-1], dtype=int),),  0,       -1j,        0], \
#       [(np.asarray([-1, 0, 1], dtype=int),),  0,       -1j,        0], \
#       [(np.asarray([ 1, 1, 0], dtype=int),),  0,        0,        1j], \
#       [(np.asarray([-1, 1, 0], dtype=int),),  0,        0,       -1j], \
#       [(np.asarray([-1,-1, 0], dtype=int),),  0,        0,       -1j], \
#       [(np.asarray([ 1,-1, 0], dtype=int),),  0,        0,        1j]]]
#B2 = np.asarray(B2)
#
#################################################################################
#def coefficients(irrep):
#  if irrep is 'T1':
#    return T1
#  elif irrep is 'A1':
#    return A1
#  elif irrep is 'E2':
#    return E2
#  elif irrep is 'B1':
#    return B1
#  elif irrep is 'B2':
#    return B2
#
