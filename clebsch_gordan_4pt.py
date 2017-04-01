import numpy as np
import cmath

import pandas as pd
from pandas import Series, DataFrame

################################################################################
# setup of lookup tables for subduction coefficients. ##########################
################################################################################
# copy-pasted from Dudek et al.: "S and D-wave phase shifts in isospin-2 \pi\pi 
# scattering from lattice QCD" [43], \ Supplementary Material

#TODO: In Cms-case second momentum is the first's negative. Maybe do not
# hardcode as numpy-array

sqrt2=np.sqrt(2.)
sqrt3=np.sqrt(3.)

################################################################################
# T1

#T1 = DataFrame({'p' : [np.array( (( 1,  0,  0), (-1,  0,  0)) ), \
#                       np.array( (( 0,  1,  0), ( 0, -1,  0)) ), \
#                       np.array( (( 0,  0,  1), ( 0,  0, -1)) ), \
#                       np.array( ((-1,  0,  0), ( 1,  0,  0)) ), \
#                       np.array( (( 0, -1,  0), ( 0,  1,  0)) ), \
#                       np.array( (( 0,  0, -1), ( 0,  0,  1)) ), \
#                       np.array( (( 1,  1,  0), (-1, -1,  0)) ), \
#                       np.array( (( 0,  1,  1), ( 0, -1, -1)) ), \
#                       np.array( (( 1,  0,  1), (-1,  0, -1)) ), \
#                       np.array( (( 1, -1,  0), (-1,  1,  0)) ), \
#                       np.array( (( 0,  1, -1), ( 0, -1,  1)) ), \
#                       np.array( ((-1,  0,  1), ( 1,  0, -1)) ), \
#                       np.array( ((-1,  1,  0), ( 1, -1,  0)) ), \
#                       np.array( (( 0, -1,  1), ( 0,  1, -1)) ), \
#                       np.array( (( 1,  0, -1), (-1,  0,  1)) ), \
#                       np.array( ((-1, -1,  0), ( 1,  1,  0)) ), \
#                       np.array( (( 0, -1, -1), ( 0,  1,  1)) ), \
#                       np.array( ((-1,  0, -1), ( 1,  0,  1)) )]*3, \
T1 = DataFrame({'p' : [( (( 1,  0,  0), (-1,  0,  0)) ), \
                       ( (( 0,  1,  0), ( 0, -1,  0)) ), \
                       ( (( 0,  0,  1), ( 0,  0, -1)) ), \
                       ( ((-1,  0,  0), ( 1,  0,  0)) ), \
                       ( (( 0, -1,  0), ( 0,  1,  0)) ), \
                       ( (( 0,  0, -1), ( 0,  0,  1)) ), \
                       ( (( 1,  1,  0), (-1, -1,  0)) ), \
                       ( (( 0,  1,  1), ( 0, -1, -1)) ), \
                       ( (( 1,  0,  1), (-1,  0, -1)) ), \
                       ( (( 1, -1,  0), (-1,  1,  0)) ), \
                       ( (( 0,  1, -1), ( 0, -1,  1)) ), \
                       ( ((-1,  0,  1), ( 1,  0, -1)) ), \
                       ( ((-1,  1,  0), ( 1, -1,  0)) ), \
                       ( (( 0, -1,  1), ( 0,  1, -1)) ), \
                       ( (( 1,  0, -1), (-1,  0,  1)) ), \
                       ( ((-1, -1,  0), ( 1,  1,  0)) ), \
                       ( (( 0, -1, -1), ( 0,  1,  1)) ), \
                       ( ((-1,  0, -1), ( 1,  0,  1)) )]*3, \
                '|J, M\rangle' : ["|0, 0\rangle"]*54, \
                'cg-coefficient' : [-1./2, \
                                           -1j/2, \
                                       0., \
                                       1./2, \
                                            1j/2, \
                                       0., \
                                      -1./4-1j/4, \
                                           -1j/4, \
                                      -1./4, \
                                      -1./4+1j/4, \
                                           -1j/4, \
                                       1./4, \
                                       1./4-1j/4, \
                                            1j/4, \
                                      -1./4, \
                                       1./4+1j/4, \
                                            1j/4, \
                                       1./4, \
                                       0., \
                                       0., \
                                       1./sqrt2, \
                                       0., \
                                       0., \
                                      -1./sqrt2, \
                                       0., \
                                       1./(2*sqrt2), \
                                       1./(2*sqrt2), \
                                       0., \
                                      -1./(2*sqrt2), \
                                       1./(2*sqrt2), \
                                       0., \
                                       1./(2*sqrt2), \
                                      -1./(2*sqrt2), \
                                       0., \
                                      -1./(2*sqrt2), \
                                      -1./(2*sqrt2), \
                                       1./2, \
                                           -1j/2, \
                                       0., \
                                      -1./2, \
                                            1j/2, \
                                       0., \
                                       1./4-1j/4, \
                                           -1j/4, \
                                       1./4, \
                                       1./4+1j/4, \
                                           -1j/4, \
                                      -1./4, \
                                      -1./4-1j/4, \
                                            1j/4, \
                                       1./4, \
                                      -1./4+1j/4, \
                                            1j/4, \
                                      -1./4]}, \
                index=pd.Index([1]*18+[2]*18+[3]*18, name='\mu'))

print T1
#T1 = [[[(np.asarray([ 1,  0,  0], dtype=int), np.asarray([-1,  0,  0], dtype=int)), -1./2], \
#       [(np.asarray([ 0,  1,  0], dtype=int), np.asarray([ 0, -1,  0], dtype=int)),      -1j/2], \
#       [(np.asarray([ 0,  0,  1], dtype=int), np.asarray([ 0,  0, -1], dtype=int)),  0.], \
#       [(np.asarray([-1,  0,  0], dtype=int), np.asarray([ 1,  0,  0], dtype=int)),  1./2], \
#       [(np.asarray([ 0, -1,  0], dtype=int), np.asarray([ 0,  1,  0], dtype=int)),       1j/2], \
#       [(np.asarray([ 0,  0, -1], dtype=int), np.asarray([ 0,  0,  1], dtype=int)),  0.], \
#       [(np.asarray([ 1,  1,  0], dtype=int), np.asarray([-1, -1,  0], dtype=int)), -1./4-1j/4], \
#       [(np.asarray([ 0,  1,  1], dtype=int), np.asarray([ 0, -1, -1], dtype=int)),      -1j/4], \
#       [(np.asarray([ 1,  0,  1], dtype=int), np.asarray([-1,  0, -1], dtype=int)), -1./4], \
#       [(np.asarray([ 1, -1,  0], dtype=int), np.asarray([-1,  1,  0], dtype=int)), -1./4+1j/4], \
#       [(np.asarray([ 0,  1, -1], dtype=int), np.asarray([ 0, -1,  1], dtype=int)),      -1j/4], \
#       [(np.asarray([-1,  0,  1], dtype=int), np.asarray([ 1,  0, -1], dtype=int)),  1./4], \
#       [(np.asarray([-1,  1,  0], dtype=int), np.asarray([ 1, -1,  0], dtype=int)),  1./4-1j/4], \
#       [(np.asarray([ 0, -1,  1], dtype=int), np.asarray([ 0,  1, -1], dtype=int)),       1j/4], \
#       [(np.asarray([ 1,  0, -1], dtype=int), np.asarray([-1,  0,  1], dtype=int)), -1./4], \
#       [(np.asarray([-1, -1,  0], dtype=int), np.asarray([ 1,  1,  0], dtype=int)),  1./4+1j/4], \
#       [(np.asarray([ 0, -1, -1], dtype=int), np.asarray([ 0,  1,  1], dtype=int)),       1j/4], \
#       [(np.asarray([-1,  0, -1], dtype=int), np.asarray([ 1,  0,  1], dtype=int)),  1./4]], \
#                                                                               \
#      [[(np.asarray([ 1,  0,  0], dtype=int), np.asarray([-1,  0,  0], dtype=int)),  0.], \
#       [(np.asarray([ 0,  1,  0], dtype=int), np.asarray([ 0, -1,  0], dtype=int)),  0.], \
#       [(np.asarray([ 0,  0,  1], dtype=int), np.asarray([ 0,  0, -1], dtype=int)),  1./sqrt2], \
#       [(np.asarray([-1,  0,  0], dtype=int), np.asarray([ 1,  0,  0], dtype=int)),  0.], \
#       [(np.asarray([ 0, -1,  0], dtype=int), np.asarray([ 0,  1,  0], dtype=int)),  0.], \
#       [(np.asarray([ 0,  0, -1], dtype=int), np.asarray([ 0,  0,  1], dtype=int)), -1./sqrt2], \
#       [(np.asarray([ 1,  1,  0], dtype=int), np.asarray([-1, -1,  0], dtype=int)),  0.], \
#       [(np.asarray([ 0,  1,  1], dtype=int), np.asarray([ 0, -1, -1], dtype=int)),  1./(2*sqrt2)], \
#       [(np.asarray([ 1,  0,  1], dtype=int), np.asarray([-1,  0, -1], dtype=int)),  1./(2*sqrt2)], \
#       [(np.asarray([ 1, -1,  0], dtype=int), np.asarray([-1,  1,  0], dtype=int)),  0.], \
#       [(np.asarray([ 0,  1, -1], dtype=int), np.asarray([ 0, -1,  1], dtype=int)), -1./(2*sqrt2)], \
#       [(np.asarray([-1,  0,  1], dtype=int), np.asarray([ 1,  0, -1], dtype=int)),  1./(2*sqrt2)], \
#       [(np.asarray([-1,  1,  0], dtype=int), np.asarray([ 1, -1,  0], dtype=int)),  0.], \
#       [(np.asarray([ 0, -1,  1], dtype=int), np.asarray([ 0,  1, -1], dtype=int)),  1./(2*sqrt2)], \
#       [(np.asarray([ 1,  0, -1], dtype=int), np.asarray([-1,  0,  1], dtype=int)), -1./(2*sqrt2)], \
#       [(np.asarray([-1, -1,  0], dtype=int), np.asarray([ 1,  1,  0], dtype=int)),  0.], \
#       [(np.asarray([ 0, -1, -1], dtype=int), np.asarray([ 0,  1,  1], dtype=int)), -1./(2*sqrt2)], \
#       [(np.asarray([-1,  0, -1], dtype=int), np.asarray([ 1,  0,  1], dtype=int)), -1./(2*sqrt2)]], \
#                                                                               \
#      [[(np.asarray([ 1,  0,  0], dtype=int), np.asarray([-1,  0,  0], dtype=int)),  1./2], \
#       [(np.asarray([ 0,  1,  0], dtype=int), np.asarray([ 0, -1,  0], dtype=int)),      -1j/2], \
#       [(np.asarray([ 0,  0,  1], dtype=int), np.asarray([ 0,  0, -1], dtype=int)),  0.], \
#       [(np.asarray([-1,  0,  0], dtype=int), np.asarray([ 1,  0,  0], dtype=int)), -1./2], \
#       [(np.asarray([ 0, -1,  0], dtype=int), np.asarray([ 0,  1,  0], dtype=int)),       1j/2], \
#       [(np.asarray([ 0,  0, -1], dtype=int), np.asarray([ 0,  0,  1], dtype=int)),  0.], \
#       [(np.asarray([ 1,  1,  0], dtype=int), np.asarray([-1, -1,  0], dtype=int)),  1./4-1j/4], \
#       [(np.asarray([ 0,  1,  1], dtype=int), np.asarray([ 0, -1, -1], dtype=int)),      -1j/4], \
#       [(np.asarray([ 1,  0,  1], dtype=int), np.asarray([-1,  0, -1], dtype=int)),  1./4], \
#       [(np.asarray([ 1, -1,  0], dtype=int), np.asarray([-1,  1,  0], dtype=int)),  1./4+1j/4], \
#       [(np.asarray([ 0,  1, -1], dtype=int), np.asarray([ 0, -1,  1], dtype=int)),      -1j/4], \
#       [(np.asarray([-1,  0,  1], dtype=int), np.asarray([ 1,  0, -1], dtype=int)), -1./4], \
#       [(np.asarray([-1,  1,  0], dtype=int), np.asarray([ 1, -1,  0], dtype=int)), -1./4-1j/4], \
#       [(np.asarray([ 0, -1,  1], dtype=int), np.asarray([ 0,  1, -1], dtype=int)),       1j/4], \
#       [(np.asarray([ 1,  0, -1], dtype=int), np.asarray([-1,  0,  1], dtype=int)),  1./4], \
#       [(np.asarray([-1, -1,  0], dtype=int), np.asarray([ 1,  1,  0], dtype=int)), -1./4+1j/4], \
#       [(np.asarray([ 0, -1, -1], dtype=int), np.asarray([ 0,  1,  1], dtype=int)),       1j/4], \
#       [(np.asarray([-1,  0, -1], dtype=int), np.asarray([ 1,  0,  1], dtype=int)), -1./4]]]
#T1 = np.asarray(T1)

################################################################################
# possibly get a sign if j11+j2-J odd and momenta are interchanged for one momentum 0
A1 = [[[(np.asarray([ 0,  0,  0], dtype=int), np.asarray([ 0,  0,  1], dtype=int)),  1.], \
       [(np.asarray([ 0,  0,  1], dtype=int), np.asarray([ 0,  0,  0], dtype=int)), -1.], \
       \
       [(np.asarray([ 1,  0,  0], dtype=int), np.asarray([-1,  0,  1], dtype=int)),  1./2], \
       [(np.asarray([ 0,  1,  0], dtype=int), np.asarray([ 0, -1,  1], dtype=int)),  1./2], \
       [(np.asarray([-1,  0,  0], dtype=int), np.asarray([ 1,  0,  1], dtype=int)),  1./2], \
       [(np.asarray([ 0, -1,  0], dtype=int), np.asarray([ 0,  1,  1], dtype=int)),  1./2], \
       [(np.asarray([-1,  0,  1], dtype=int), np.asarray([ 1,  0,  0], dtype=int)), -1./2], \
       [(np.asarray([ 0, -1,  1], dtype=int), np.asarray([ 0,  1,  0], dtype=int)), -1./2], \
       [(np.asarray([ 1,  0,  1], dtype=int), np.asarray([-1,  0,  0], dtype=int)), -1./2], \
       [(np.asarray([ 0,  1,  1], dtype=int), np.asarray([ 0, -1,  0], dtype=int)), -1./2], \
       \
       [(np.asarray([ 1,  1,  0], dtype=int), np.asarray([-1, -1,  1], dtype=int)),  1./2], \
       [(np.asarray([ 1, -1,  0], dtype=int), np.asarray([-1,  1,  1], dtype=int)),  1./2], \
       [(np.asarray([-1,  1,  0], dtype=int), np.asarray([ 1, -1,  1], dtype=int)),  1./2], \
       [(np.asarray([-1, -1,  0], dtype=int), np.asarray([ 1,  1,  1], dtype=int)),  1./2], \
       [(np.asarray([-1, -1,  1], dtype=int), np.asarray([ 1,  1,  0], dtype=int)), -1./2], \
       [(np.asarray([-1,  1,  1], dtype=int), np.asarray([ 1, -1,  0], dtype=int)), -1./2], \
       [(np.asarray([ 1, -1,  1], dtype=int), np.asarray([-1,  1,  0], dtype=int)), -1./2], \
       [(np.asarray([ 1,  1,  1], dtype=int), np.asarray([-1, -1,  0], dtype=int)), -1./2], \
       \
       [(np.asarray([ 0,  0, -1], dtype=int), np.asarray([ 0,  0,  2], dtype=int)),  1.], \
       [(np.asarray([ 0,  0,  2], dtype=int), np.asarray([ 0,  0, -1], dtype=int)), -1.], \
       \
       [(np.asarray([ 0,  0,  0], dtype=int), np.asarray([ 0,  1,  1], dtype=int)),  1.], \
       [(np.asarray([ 0,  1,  1], dtype=int), np.asarray([ 0,  0,  0], dtype=int)), -1.], \
       \
#       ([np.asarray([ 0,  1,  0], dtype=int), np.asarray([ 0,  0,  1], dtype=int)),  1./sqrt2], \
#       ([np.asarray([ 0,  0,  1], dtype=int), np.asarray([ 0,  1,  0], dtype=int)),  1./sqrt2], \
#       \
       [(np.asarray([ 1,  0,  0], dtype=int), np.asarray([-1,  1,  1], dtype=int)),  1./sqrt2], \
       [(np.asarray([-1,  0,  0], dtype=int), np.asarray([ 1,  1,  1], dtype=int)),  1./sqrt2], \
       [(np.asarray([-1,  1,  1], dtype=int), np.asarray([ 1,  0,  0], dtype=int)), -1./sqrt2], \
       [(np.asarray([ 1,  1,  1], dtype=int), np.asarray([-1,  0,  0], dtype=int)), -1./sqrt2], \
       \
#       ([np.asarray([ 1,  1,  0], dtype=int), np.asarray([-1,  0,  1], dtype=int)),  1./2], \
#       ([np.asarray([ 1,  0,  1], dtype=int), np.asarray([-1,  1,  0], dtype=int)),  1./2], \
#       ([np.asarray([-1,  0,  1], dtype=int), np.asarray([ 1,  1,  0], dtype=int)),  1./2], \
#       ([np.asarray([-1,  1,  0], dtype=int), np.asarray([ 1,  0,  1], dtype=int)),  1./2], \
#       \
       [(np.asarray([ 0,  1, -1], dtype=int), np.asarray([ 0,  0,  2], dtype=int)),  1./sqrt2], \
       [(np.asarray([ 0, -1,  1], dtype=int), np.asarray([ 0,  2,  0], dtype=int)),  1./sqrt2], \
       [(np.asarray([ 0,  0,  2], dtype=int), np.asarray([ 0,  1, -1], dtype=int)), -1./sqrt2], \
       [(np.asarray([ 0,  2,  0], dtype=int), np.asarray([ 0, -1,  1], dtype=int)), -1./sqrt2], \
       \
       [(np.asarray([ 0,  0,  0], dtype=int), np.asarray([ 1,  1,  1], dtype=int)),  1.], \
       [(np.asarray([ 1,  1,  1], dtype=int), np.asarray([ 0,  0,  0], dtype=int)), -1.], \
       \
       [(np.asarray([ 1,  0,  0], dtype=int), np.asarray([ 0,  1,  1], dtype=int)),  1./sqrt3], \
       [(np.asarray([ 0,  1,  0], dtype=int), np.asarray([ 1,  0,  1], dtype=int)),  1./sqrt3], \
       [(np.asarray([ 0,  0,  1], dtype=int), np.asarray([ 1,  1,  0], dtype=int)),  1./sqrt3], \
       [(np.asarray([ 0,  1,  1], dtype=int), np.asarray([ 1,  0,  0], dtype=int)), -1./sqrt3], \
       [(np.asarray([ 1,  0,  1], dtype=int), np.asarray([ 0,  1,  0], dtype=int)), -1./sqrt3], \
       [(np.asarray([ 1,  1,  0], dtype=int), np.asarray([ 0,  0,  1], dtype=int)), -1./sqrt3], \
       \
       [(np.asarray([ 2,  0,  0], dtype=int), np.asarray([-1,  1,  1], dtype=int)), (-1)* 1./sqrt3], \
       [(np.asarray([ 0,  2,  0], dtype=int), np.asarray([ 1, -1,  1], dtype=int)), (-1)* 1./sqrt3], \
       [(np.asarray([ 0,  0,  2], dtype=int), np.asarray([ 1,  1, -1], dtype=int)), (-1)* 1./sqrt3], \
       [(np.asarray([-1,  1,  1], dtype=int), np.asarray([ 2,  0,  0], dtype=int)), (-1)*(-1./sqrt3)], \
       [(np.asarray([ 1, -1,  1], dtype=int), np.asarray([ 0,  2,  0], dtype=int)), (-1)*(-1./sqrt3)], \
       [(np.asarray([ 1,  1, -1], dtype=int), np.asarray([ 0,  0,  2], dtype=int)), (-1)*(-1./sqrt3)], \
       \
       [(np.asarray([ 0,  0,  0], dtype=int), np.asarray([ 0,  0,  2], dtype=int)),  1.], \
       [(np.asarray([ 0,  0,  2], dtype=int), np.asarray([ 0,  0,  0], dtype=int)), -1.]]]
A1 = np.asarray(A1)

################################################################################
E2 = [[[(np.asarray([ 1,  0,  0], dtype=int), np.asarray([-1,  0,  1], dtype=int)),  0], \
       [(np.asarray([ 0,  1,  0], dtype=int), np.asarray([ 0, -1,  1], dtype=int)),   -1j/sqrt2], \
       [(np.asarray([-1,  0,  0], dtype=int), np.asarray([ 1,  0,  1], dtype=int)),  0], \
       [(np.asarray([ 0, -1,  0], dtype=int), np.asarray([ 0,  1,  1], dtype=int)),    1j/sqrt2], \
       [(np.asarray([-1,  0,  1], dtype=int), np.asarray([ 1,  0,  0], dtype=int)),  0], \
       [(np.asarray([ 0, -1,  1], dtype=int), np.asarray([ 0,  1,  0], dtype=int)),    1j/sqrt2], \
       [(np.asarray([ 1,  0,  1], dtype=int), np.asarray([-1,  0,  0], dtype=int)),  0], \
       [(np.asarray([ 0,  1,  1], dtype=int), np.asarray([ 0, -1,  0], dtype=int)),   -1j/sqrt2], \
       \
       [(np.asarray([ 1,  1,  0], dtype=int), np.asarray([-1, -1,  1], dtype=int)),  -1j/2], \
       [(np.asarray([ 1, -1,  0], dtype=int), np.asarray([-1,  1,  1], dtype=int)),   1j/2], \
       [(np.asarray([-1,  1,  0], dtype=int), np.asarray([ 1, -1,  1], dtype=int)),  -1j/2], \
       [(np.asarray([-1, -1,  0], dtype=int), np.asarray([ 1,  1,  1], dtype=int)),   1j/2], \
       [(np.asarray([-1, -1,  1], dtype=int), np.asarray([ 1,  1,  0], dtype=int)),   1j/2], \
       [(np.asarray([-1,  1,  1], dtype=int), np.asarray([ 1, -1,  0], dtype=int)),  -1j/2], \
       [(np.asarray([ 1, -1,  1], dtype=int), np.asarray([-1,  1,  0], dtype=int)),   1j/2], \
       [(np.asarray([ 1,  1,  1], dtype=int), np.asarray([-1, -1,  0], dtype=int)),  -1j/2]], \
                                                                              \
      [[(np.asarray([ 1,  0,  0], dtype=int), np.asarray([-1,  0,  1], dtype=int)), -1./sqrt2], \
       [(np.asarray([ 0,  1,  0], dtype=int), np.asarray([ 0, -1,  1], dtype=int)),  0], \
       [(np.asarray([-1,  0,  0], dtype=int), np.asarray([ 1,  0,  1], dtype=int)),  1./sqrt2], \
       [(np.asarray([ 0, -1,  0], dtype=int), np.asarray([ 0,  1,  1], dtype=int)),  0], \
       [(np.asarray([-1,  0,  1], dtype=int), np.asarray([ 1,  0,  0], dtype=int)),  1./sqrt2], \
       [(np.asarray([ 0, -1,  1], dtype=int), np.asarray([ 0,  1,  0], dtype=int)),  0], \
       [(np.asarray([ 1,  0,  1], dtype=int), np.asarray([-1,  0,  0], dtype=int)), -1./sqrt2], \
       [(np.asarray([ 0,  1,  1], dtype=int), np.asarray([ 0, -1,  0], dtype=int)),  0], \
       \
       [(np.asarray([ 1,  1,  0], dtype=int), np.asarray([-1, -1,  1], dtype=int)), -1./2], \
       [(np.asarray([ 1, -1,  0], dtype=int), np.asarray([-1,  1,  1], dtype=int)), -1./2], \
       [(np.asarray([-1,  1,  0], dtype=int), np.asarray([ 1, -1,  1], dtype=int)),  1./2], \
       [(np.asarray([-1, -1,  0], dtype=int), np.asarray([ 1,  1,  1], dtype=int)),  1./2], \
       [(np.asarray([-1, -1,  1], dtype=int), np.asarray([ 1,  1,  0], dtype=int)),  1./2], \
       [(np.asarray([-1,  1,  1], dtype=int), np.asarray([ 1, -1,  0], dtype=int)),  1./2], \
       [(np.asarray([ 1, -1,  1], dtype=int), np.asarray([-1,  1,  0], dtype=int)), -1./2], \
       [(np.asarray([ 1,  1,  1], dtype=int), np.asarray([-1, -1,  0], dtype=int)), -1./2]]]
E2 = np.asarray(E2)
################################################################################

B1 = [[[np.asarray([ 0,  1,  0], dtype=int), np.asarray([ 0,  0,  1], dtype=int),  1j/sqrt2], \
       [np.asarray([ 0,  0,  1], dtype=int), np.asarray([ 0,  1,  0], dtype=int), -1j/sqrt2], \
       \
       [np.asarray([ 1,  1,  0], dtype=int), np.asarray([-1,  0,  1], dtype=int),  1j/2], \
       [np.asarray([ 1,  0,  1], dtype=int), np.asarray([-1,  1,  0], dtype=int), -1j/2], \
       [np.asarray([-1,  0,  1], dtype=int), np.asarray([ 1,  1,  0], dtype=int), -1j/2], \
       [np.asarray([-1,  1,  0], dtype=int), np.asarray([ 1,  0,  1], dtype=int),  1j/2], \
       \
       [np.asarray([ 0,  1, -1], dtype=int), np.asarray([ 0,  0,  2], dtype=int),  1j/sqrt2], \
       [np.asarray([ 0, -1,  1], dtype=int), np.asarray([ 0,  2,  0], dtype=int), -1j/sqrt2], \
       [np.asarray([ 0,  0,  2], dtype=int), np.asarray([ 0,  1, -1], dtype=int), -1j/sqrt2], \
       [np.asarray([ 0,  2,  0], dtype=int), np.asarray([ 0, -1,  1], dtype=int),  1j/sqrt2]]]

B1 = np.asarray(B1)
################################################################################

B2 = [[[np.asarray([ 1,  0,  0], dtype=int), np.asarray([-1,  1,  1], dtype=int), (-1)*(-1./sqrt2)], \
       [np.asarray([-1,  0,  0], dtype=int), np.asarray([ 1,  1,  1], dtype=int), (-1)* 1./sqrt2], \
       [np.asarray([-1,  1,  1], dtype=int), np.asarray([ 1,  0,  0], dtype=int), (-1)* 1./sqrt2], \
       [np.asarray([ 1,  1,  1], dtype=int), np.asarray([-1,  0,  0], dtype=int), (-1)*(-1./sqrt2)], \
       \
       [np.asarray([ 1,  1,  0], dtype=int), np.asarray([-1,  0,  1], dtype=int),  1./2], \
       [np.asarray([ 1,  0,  1], dtype=int), np.asarray([-1,  1,  0], dtype=int),  1./2], \
       [np.asarray([-1,  0,  1], dtype=int), np.asarray([ 1,  1,  0], dtype=int), -1./2], \
       [np.asarray([-1,  1,  0], dtype=int), np.asarray([ 1,  0,  1], dtype=int), -1./2]]]
B2 = np.asarray(B2)
################################################################################

def coefficients(irrep):
  if irrep is 'T1':
    return T1
  elif irrep is 'A1':
    return A1
  elif irrep is 'E2':
    return E2
  elif irrep is 'B1':
    return B1
  elif irrep is 'B2':
    return B2

